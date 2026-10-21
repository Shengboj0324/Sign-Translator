"""R2Store.get retries a body stream that breaks mid-read, and raises client errors at once."""
import pytest
from botocore.exceptions import ClientError, ResponseStreamingError

from signtranslator.data.text_motion_dataset import R2Store


class Body:
    def __init__(self, fail, error=None):
        self.fail, self.error = fail, error
    def read(self):
        if self.fail:
            raise self.error or ResponseStreamingError(error=BrokenPipeError(32, "Broken pipe"))
        return b"payload"


class Client:
    def __init__(self, failures, missing=False, error=None):
        self.failures, self.missing, self.calls, self.error = failures, missing, 0, error
    def get_object(self, Bucket, Key):
        self.calls += 1
        if self.missing:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": Body(self.calls <= self.failures, self.error)}


def store(client, monkeypatch):
    s = R2Store({"R2_ENDPOINT_URL": "https://x", "R2_BUCKET": "b", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"})
    s._client = client
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return s


def test_a_broken_body_stream_is_retried(monkeypatch):
    client = Client(failures=2)
    assert store(client, monkeypatch).get("k") == b"payload" and client.calls == 3


def test_persistent_breaks_and_client_errors_are_raised(monkeypatch):
    with pytest.raises(ResponseStreamingError):
        store(Client(failures=99), monkeypatch).get("k", attempts=3)
    client = Client(failures=0, missing=True)
    with pytest.raises(ClientError):
        store(client, monkeypatch).get("k")
    assert client.calls == 1


def test_a_tls_failure_mid_body_is_retried(monkeypatch):
    from urllib3.exceptions import SSLError
    client = Client(failures=1, error=SSLError("[SSL] record layer failure (_ssl.c:2580)"))
    assert store(client, monkeypatch).get("k") == b"payload" and client.calls == 2


class Throttling:
    """HeadObject answers 429 with no body (so no error code) a few times, then succeeds or reports 404."""
    def __init__(self, throttled, missing=False):
        self.throttled, self.missing, self.calls = throttled, missing, 0
    def head_object(self, Bucket, Key):
        self.calls += 1
        if self.calls <= self.throttled:
            raise ClientError({"Error": {"Code": "429", "Message": "Too Many Requests"},
                               "ResponseMetadata": {"HTTPStatusCode": 429}}, "HeadObject")
        if self.missing:
            raise ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")
        return {}


def test_a_throttled_head_is_retried_and_a_missing_key_is_not(monkeypatch):
    client = Throttling(throttled=3)
    assert store(client, monkeypatch).exists("k") and client.calls == 4
    client = Throttling(throttled=2, missing=True)
    assert not store(client, monkeypatch).exists("k") and client.calls == 3
    with pytest.raises(ClientError):
        store(Throttling(throttled=99), monkeypatch).exists("k")
