import pytest

from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (LoadBalancingRouter, RoundRobinRouter,
                                       create_router)
from tensorrt_llm.logger import logger
import threading
import time
import asyncio
from unittest import mock

# Mock class for metadata server
class MockMetadataServer:
    """Mock metadata server for testing router interactions"""
    
    def __init__(self):
        self.servers = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            return self.servers.get(key)
    
    def put(self, key, value):
        with self.lock:
            self.servers[key] = value
            return True
    
    def remove(self, key):
        with self.lock:
            if key in self.servers:
                del self.servers[key]
                return True
            return False
    
    def add_server(self, key, url):
        with self.lock:
            self.servers[key] = url
            return True
            
    def keys(self, prefix=""):
        with self.lock:
            return [k for k in self.servers.keys() if k.startswith(prefix)]


@pytest.fixture
def servers():
    return ["server1", "server2", "server3"]


def get_prompt_lengths():
    return [100, 500, 10, 400, 2000, 100]


@pytest.fixture
def context_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple CompletionRequest objects with different prompts
    return [
        CompletionRequest(model="TinyLlama",
                          prompt=["the " * length],
                          disaggregated_params=DisaggregatedParams(
                              request_type="context_only",
                              first_gen_tokens=[1000],
                              ctx_request_id=str(index),
                              encoded_opaque_state=None,
                              draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def chat_context_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        ChatCompletionRequest(messages=[{
            "role": "user",
            "content": "the " * length
        }],
                              model="TinyLlama",
                              disaggregated_params=DisaggregatedParams(
                                  request_type="context_only",
                                  first_gen_tokens=[1000],
                                  ctx_request_id=str(index),
                                  encoded_opaque_state=None,
                                  draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def gen_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        CompletionRequest(model="TinyLlama",
                          prompt=["the " * length],
                          disaggregated_params=DisaggregatedParams(
                              request_type="generation_only",
                              first_gen_tokens=[1000],
                              ctx_request_id=str(index),
                              encoded_opaque_state=None,
                              draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def chat_gen_requests():

    prompt_lengths = get_prompt_lengths()

    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        ChatCompletionRequest(messages=[{
            "role": "user",
            "content": "the " * length
        }],
                              model="TinyLlama",
                              disaggregated_params=DisaggregatedParams(
                                  request_type="generation_only",
                                  first_gen_tokens=[1000],
                                  ctx_request_id=str(index),
                                  encoded_opaque_state=None,
                                  draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.mark.asyncio
async def test_round_robin_router(servers, context_requests):
    router = RoundRobinRouter(servers)
    server_sequence = [
        await router.get_next_server(req) for req in context_requests
    ]
    assert server_sequence == [
        "server1", "server2", "server3", "server1", "server2", "server3"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", [
    "context_requests", "chat_context_requests", "gen_requests",
    "chat_gen_requests"
])
async def test_request_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(servers, use_tokens=False)
    requests = request.getfixturevalue(requests_fixture)

    server = await router.get_next_server(requests[0])
    assert server == "server1"
    server = await router.get_next_server(requests[1])
    assert server == "server2"
    server = await router.get_next_server(requests[2])
    assert server == "server3"

    # Similulate terminating 3rd request (on server 3)
    await router.finish_request(requests[2])

    # Now server3 is least loaded
    server = await router.get_next_server(requests[3])
    assert server == "server3"

    # Simulate terminating 4th request (on server 3)
    await router.finish_request(requests[1])

    # Now server2 is least loaded
    server = await router.get_next_server(requests[4])
    assert server == "server2"


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", ["context_requests"])
async def test_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(servers, use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    server_sequence = [await router.get_next_server(req) for req in requests]
    # Loads at each step:
    # Step 0:
    # server1: 100
    # server2: 0
    # server3: 0

    # Step 1:
    # server1: 100
    # server2: 500
    # server3: 0

    # Step 2:
    # server1: 100
    # server2: 500
    # server3: 10

    # Step 3:
    # server1: 100
    # server2: 500
    # server3: 410

    # Step 4:
    # server1: 2100
    # server2: 500
    # server3: 410

    # Step 5:
    # server1: 2100
    # server2: 500
    # server3: 510

    assert server_sequence == [
        "server1", "server2", "server3", "server3", "server1", "server3"
    ]

    # Simulate terminating 5th request (on server 1)
    await router.finish_request(requests[4])
    server = await router.get_next_server(requests[4])

    # New loads:
    #server1: 100
    #server2: 500
    #server3: 510
    assert server == "server1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requests_fixture",
    ["chat_context_requests", "gen_requests", "chat_gen_requests"])
async def test_gen_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(servers, use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    # Should throw an error if trying to use tokens load balancing with gen-only requests or chat completion requests
    with pytest.raises(ValueError):
        await router.get_next_server(requests[0])


def test_create_router(servers):
    round_robin_router = create_router("round_robin", servers)
    assert isinstance(round_robin_router, RoundRobinRouter)

    requests_load_balancing_router = create_router("requests_load_balancing",
                                                   servers)
    assert isinstance(requests_load_balancing_router, LoadBalancingRouter)
    assert not requests_load_balancing_router._use_tokens

    tokens_load_balancing_router = create_router("tokens_load_balancing",
                                                 servers)
    assert isinstance(tokens_load_balancing_router, LoadBalancingRouter)
    assert tokens_load_balancing_router._use_tokens

    with pytest.raises(ValueError):
        create_router("unsupported_router", servers)


@pytest.fixture
def mock_metadata_server():
    return MockMetadataServer()


@pytest.mark.slow
def test_fetch_live_servers_context(mock_metadata_server):
    """Test fetching live context servers"""
    # Create router with mock metadata server
    router = RoundRobinRouter(
        server_role="context", 
        metadata_servers=[mock_metadata_server]
    )
    
    # Start server monitoring with a shorter poll interval for testing
    # but still long enough to verify the actual behavior
    poll_interval = 10  # seconds
    asyncio.run(router.start_server_monitoring(poll_interval=poll_interval))
    
    try:
        # Initial check - should be no servers
        servers = router.fetch_live_servers()
        assert len(servers) == 0, "Should have no servers initially"
        
        # Add a server
        server_key = "servers/context/server1"
        server_url = "http://localhost:8001"
        mock_metadata_server.add_server(server_key, {"url": server_url})
        
        # Wait for the polling interval to pass (add 50% buffer)
        wait_time = poll_interval * 1.5
        logger.info(f"Waiting {wait_time} seconds for server to be detected...")
        time.sleep(wait_time)
        
        # Fetch servers again
        servers = router.fetch_live_servers()
        assert len(servers) == 1, "Should have one server after adding and waiting"
        assert servers[0] == server_url, "Server URL should match what was added"
        
        # Add another server
        server_key2 = "servers/context/server2"
        server_url2 = "http://localhost:8002" 
        mock_metadata_server.add_server(server_key2, {"url": server_url2})
        
        # Wait for the polling interval again
        logger.info(f"Waiting {wait_time} seconds for second server to be detected...")
        time.sleep(wait_time)
        
        # Fetch servers again
        servers = router.fetch_live_servers()
        assert len(servers) == 2, "Should have two servers after adding second one and waiting"
        assert server_url in servers, "First server should still be present"
        assert server_url2 in servers, "Second server should be present"
        
        # Remove a server
        mock_metadata_server.remove(server_key)
        
        # Wait for the polling interval again
        logger.info(f"Waiting {wait_time} seconds for server removal to be detected...")
        time.sleep(wait_time)
        
        # Fetch servers again
        servers = router.fetch_live_servers()
        assert len(servers) == 1, "Should have one server after removing one and waiting"
        assert servers[0] == server_url2, "Remaining server should be the second one"
    finally:
        # Clean up
        asyncio.run(router.stop_server_monitoring())


@pytest.mark.slow
def test_fetch_live_servers_with_delay(mock_metadata_server):
    """Test fetching live servers with the actual polling delay"""
    # Create router with mock metadata server
    poll_interval = 5  # seconds
    
    router = RoundRobinRouter(
        server_role="context", 
        metadata_servers=[mock_metadata_server]
    )
    
    # Start server monitoring with shorter interval for testing
    asyncio.run(router.start_server_monitoring(poll_interval=poll_interval))
    
    try:
        # Initial check - should be no servers
        servers = router.fetch_live_servers()
        assert len(servers) == 0, "Should have no servers initially"
        
        # Add a server
        server_key = "servers/context/server1"
        server_url = "http://localhost:8001"
        mock_metadata_server.add_server(server_key, {"url": server_url})
        
        # Wait for a bit less than the polling interval - should still have no servers
        short_wait = poll_interval * 0.4
        logger.info(f"Waiting {short_wait} seconds (less than polling interval)...")
        time.sleep(short_wait)
        
        # Verify server isn't discovered yet
        servers = router.fetch_live_servers()
        assert len(servers) == 0, "Should still have no servers before polling interval completes"
        
        # Wait for the polling interval to pass
        remaining_wait = poll_interval * 1.2
        logger.info(f"Waiting additional {remaining_wait} seconds for server to be detected...")
        time.sleep(remaining_wait)
        
        # Now should have the server
        servers = router.fetch_live_servers()
        assert len(servers) == 1, "Should have one server after polling interval"
        assert servers[0] == server_url, "Server URL should match what was added"
        
        # Remove the server
        mock_metadata_server.remove(server_key)
        
        # Wait for polling interval to pass
        wait_time = poll_interval * 1.5
        logger.info(f"Waiting {wait_time} seconds for server removal to be detected...")
        time.sleep(wait_time)
        
        # Should now be empty again
        servers = router.fetch_live_servers()
        assert len(servers) == 0, "Should have no servers after removal and waiting"
    finally:
        # Clean up
        asyncio.run(router.stop_server_monitoring())


@pytest.mark.slow
def test_server_health_check(mock_metadata_server):
    """Test that unhealthy servers are filtered out"""
    # Create router with mock metadata server
    poll_interval = 5  # seconds
    
    router = RoundRobinRouter(
        server_role="context",
        metadata_servers=[mock_metadata_server]
    )
    
    # Start server monitoring
    asyncio.run(router.start_server_monitoring(poll_interval=poll_interval))
    
    try:
        # Add two servers
        server_key1 = "servers/context/server1"
        server_url1 = "http://localhost:8001"
        mock_metadata_server.add_server(server_key1, {"url": server_url1})
        
        server_key2 = "servers/context/server2"
        server_url2 = "http://localhost:8002"
        mock_metadata_server.add_server(server_key2, {"url": server_url2})
        
        # Wait for the polling interval to pass
        wait_time = poll_interval * 1.5
        logger.info(f"Waiting {wait_time} seconds for servers to be detected...")
        time.sleep(wait_time)
        
        # Mock the is_server_healthy method to simulate one server being down
        with mock.patch.object(router, 'is_server_healthy') as mock_is_healthy:
            # Only the second server is "healthy"
            mock_is_healthy.side_effect = lambda url: url == server_url2
            
            # Fetch servers with health check
            servers = router.fetch_live_servers(check_health=True)
            assert len(servers) == 1, "Should have one healthy server"
            assert servers[0] == server_url2, "Only healthy server should be returned"
    finally:
        # Clean up
        asyncio.run(router.stop_server_monitoring())


@pytest.mark.slow
def test_load_balancing_router_fetch_servers(mock_metadata_server):
    """Test that LoadBalancingRouter fetches servers correctly"""
    # Create router with mock metadata server
    poll_interval = 10  # seconds
    
    router = LoadBalancingRouter(
        server_role="context",
        metadata_servers=[mock_metadata_server]
    )
    
    # Start server monitoring
    asyncio.run(router.start_server_monitoring(poll_interval=poll_interval))
    
    try:
        # Add two servers
        server_key1 = "servers/context/server1"
        server_url1 = "http://localhost:8001"
        mock_metadata_server.add_server(server_key1, {"url": server_url1})
        
        server_key2 = "servers/context/server2"
        server_url2 = "http://localhost:8002"
        mock_metadata_server.add_server(server_key2, {"url": server_url2})
        
        # Wait for the polling interval to pass
        wait_time = poll_interval * 1.5
        logger.info(f"Waiting {wait_time} seconds for servers to be detected...")
        time.sleep(wait_time)
        
        # Fetch servers
        servers = router.fetch_live_servers()
        assert len(servers) == 2, "Should have two servers after waiting"
        
        # Remove all servers
        mock_metadata_server.remove(server_key1)
        mock_metadata_server.remove(server_key2)
        
        # Wait for the polling interval to pass
        logger.info(f"Waiting {wait_time} seconds for server removals to be detected...")
        time.sleep(wait_time)
        
        # Test handling of no servers
        servers = router.fetch_live_servers()
        assert len(servers) == 0, "Should have no servers after removing all and waiting"
        
        # Test get_next_server with no servers should raise ValueError
        with pytest.raises(ValueError):
            router.get_next_server()
    finally:
        # Clean up
        asyncio.run(router.stop_server_monitoring())
