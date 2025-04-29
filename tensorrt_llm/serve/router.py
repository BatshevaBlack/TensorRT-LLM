import asyncio
import heapq
import os
import socket
import json
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metadata_server import JsonDictionary
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)


def get_request_num_tokens(
        request: Union[CompletionRequest, ChatCompletionRequest]) -> int:
    if request.disaggregated_params.request_type == "context_only":
        if isinstance(request, ChatCompletionRequest):
            raise ValueError(
                "LoadBalancing router with tokens doesn't support ChatCompletionRequest yet"
            )

        if isinstance(request.prompt, str) or \
            (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        num_tokens = sum(len(prompt) for prompt in prompts)
    elif request.disaggregated_params.request_type == "generation_only":
        raise ValueError(
            "LoadBalancing router with tokens doesn't support generation_only requests"
        )
    else:
        raise ValueError(
            f"Unsupported request type: {request.disaggregated_params.request_type}"
        )

    return num_tokens


class ServerState:

    def __init__(self, server: str, use_tokens: bool = False):
        self._server = server
        self._num_active_requests = 0
        self._num_active_tokens = 0
        self._use_tokens = use_tokens
        self._lock = asyncio.Lock()

    async def increment_load(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests += 1
            self._num_active_tokens += num_tokens

    async def decrement_load(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        num_tokens = get_request_num_tokens(request) if self._use_tokens else 0
        async with self._lock:
            self._num_active_requests -= 1
            self._num_active_tokens -= num_tokens

    async def is_healthy(self) -> bool:
        try:
            async with self._session.get(self._server + "/health") as response:
                return response.status == 200
        except Exception:
            return False


class Router(ABC):

    def __init__(self,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None):
        self._servers = servers or []
        self._metadata_server = metadata_server
        self._monitor_task = None
        self._lock = asyncio.Lock()
        self._session = None
        # Default timeout for health checks (in seconds)
        self._health_check_timeout = 5.0
        logger.debug(f"Initialized Router with default health check timeout: {self._health_check_timeout}s")
        
        # Create a persistent session for health checks
        self._init_session()
    
    def _init_session(self):
        """Initialize the HTTP session for health checks"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
    async def close_session(self):
        """Close the HTTP session"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    @abstractmethod
    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        pass

    @abstractmethod
    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass

    async def start_server_monitoring(self, poll_interval: int = 10):
        """Start monitoring servers update from metadata service"""
        self._monitor_task = asyncio.create_task(
            self._monitor_servers(poll_interval))

    async def stop_server_monitoring(self):
        """Stop monitoring servers update from metadata service"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        # Close session when stopping monitoring
        await self.close_session()

    async def _monitor_servers(self, poll_interval: int = 10):
        """Monitor servers update from metadata service"""
        while True:
            if self._metadata_server:
                # Get servers from metadata
                server_key_map = await self.fetch_live_servers()
                
                # Check health and get live servers
                live_servers = await self.check_servers_health(server_key_map)
                
                async with self._lock:
                    if live_servers != self._servers:
                        self._servers = live_servers

            await asyncio.sleep(poll_interval)

    async def fetch_live_servers(self) -> Dict[str, str]:
        """Fetch all servers from metadata service and return {url: key} mapping"""
        if not self._metadata_server:
            return {server: "" for server in self._servers}

        server_key_map = {}
        try:
            # Get all keys from the metadata server
            all_keys = self._metadata_server.keys()

            # Filter keys that start with 'trtllm/' and extract server metadata
            for key in all_keys:
                if key.startswith('trtllm/'):
                    server_metadata = self._metadata_server.get(key)
                    if server_metadata and isinstance(server_metadata, dict) and 'url' in server_metadata:
                        server_key_map[server_metadata['url']] = key
                        
                        # Check if metadata includes health check timeout
                        if 'health_check_timeout' in server_metadata:
                            try:
                                self._health_check_timeout = float(server_metadata['health_check_timeout'])
                                logger.debug(f"Using health check timeout: {self._health_check_timeout}s")
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid health_check_timeout value: {server_metadata['health_check_timeout']}")
            
        except Exception as e:
            logger.error(f"Error fetching servers from metadata service: {e}")

        return server_key_map

    async def check_servers_health(self, server_key_map: Dict[str, str]) -> List[str]:
        """Check health of servers and remove dead ones from metadata service"""
        live_servers = []
        dead_servers = []
        
        try:
            # Check health of each server
            for server_url in server_key_map:
                # First attempt - no printing errors
                is_healthy = await self._check_server_health(server_url, silent=True)
                
                # If first attempt failed, try again before declaring server dead
                if not is_healthy:
                    # Second attempt - will print errors if it fails
                    is_healthy = await self._check_server_health(server_url, silent=False)
                    
                    if not is_healthy:
                        # Only now add to dead servers
                        dead_servers.append(server_url)
                        logger.warning(f"Server {server_url} is not healthy after retry - removing")
                    else:
                        live_servers.append(server_url)
                else:
                    live_servers.append(server_url)
            
            # Remove dead servers from etcd
            for dead_server in dead_servers:
                etcd_key = server_key_map.get(dead_server)
                if etcd_key:
                    try:
                        logger.info(f"Removing dead server {dead_server} from metadata server")
                        self._metadata_server.remove(etcd_key)
                    except Exception as e:
                        logger.error(f"Error removing dead server from metadata service: {e}")
            
        except Exception as e:
            logger.error(f"Error checking server health: {e}")

        return live_servers if live_servers else self._servers
        
    async def _check_server_health(self, server_url, silent=False) -> bool:
        """Check if a server is healthy by querying its health endpoint"""
        try:
            async with self._session.get(f"{server_url}/health", timeout=self._health_check_timeout) as response:
                if response.status != 200:
                    if not silent:
                        logger.warning(f"Server {server_url} is not healthy (status: {response.status})")
                    return False
                return True
        except Exception as e:
            if not silent:
                logger.warning(f"Server {server_url} is not reachable: {e}")
            return False


class RoundRobinRouter(Router):

    def __init__(self,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None):
        super().__init__(servers, metadata_server)
        self._server_idx = 0

    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        server = self._servers[self._server_idx]
        self._server_idx = (self._server_idx + 1) % len(self._servers)
        return server

    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        pass


class LoadBalancingRouter(Router):

    def __init__(self,
                 servers: List[str] = None,
                 metadata_server: JsonDictionary = None,
                 use_tokens: bool = False):
        super().__init__(servers, metadata_server)
        self._lock = asyncio.Lock()
        # Load map between servers and their number of tokens processed
        self._server_state = {}
        self._server_load_heap = []

        # Routing table to map requests to servers
        self._req_routing_table = {}

        self._use_tokens = use_tokens
        self._init_heap()

    def _init_heap(self):
        for server in self._servers:
            self._server_state[server] = ServerState(server, self._use_tokens)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

    async def get_next_server(
            self, request: Union[CompletionRequest,
                                 ChatCompletionRequest]) -> str:
        async with self._lock:
            server = heapq.heappop(self._server_load_heap)[1]
            await self._server_state[server].increment_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))

            self._req_routing_table[id(request)] = server

        return server

    def _get_server_load(self, server):
        return self._server_state[server]._num_active_tokens if self._use_tokens \
            else self._server_state[server]._num_active_requests

    async def finish_request(self, request: Union[CompletionRequest,
                                                  ChatCompletionRequest]):
        async with self._lock:
            server = self._req_routing_table[id(request)]
            await self._server_state[server].decrement_load(request)
            heapq.heappush(self._server_load_heap,
                           (self._get_server_load(server), server))
            del self._req_routing_table[id(request)]


def create_router(router_type: str,
                  servers: List[str],
                  metadata_server: JsonDictionary = None) -> Router:
    """
    Factory function to create different types of router instances.

    Args:
        router_type (str): Type of router to create. Supported values:
            - "round_robin": Creates a RoundRobinRouter
            - "requests_load_balancing": Creates a LoadBalancingRouter, which balances requests across instances
            - "tokens_load_balancing": Creates a LoadBalancingRouter, which balances tokens across instances
        servers: List of server URLs

    Returns:
        Router: An instance of the requested router type

    Raises:
        ValueError: If an unsupported router type is provided
    """

    router_map = {
        "round_robin": RoundRobinRouter,
        "requests_load_balancing": LoadBalancingRouter,
        "tokens_load_balancing": LoadBalancingRouter
    }

    router_class = router_map.get(router_type.lower())
    if router_class is None:
        raise ValueError(f"Unsupported router type: {router_type}. "
                         f"Supported types are: {list(router_map.keys())}")

    if router_type.endswith("load_balancing"):
        use_tokens = True if router_type.startswith("tokens") else False
        return router_class(servers, metadata_server, use_tokens=use_tokens)
    else:
        return router_class(servers, metadata_server)
