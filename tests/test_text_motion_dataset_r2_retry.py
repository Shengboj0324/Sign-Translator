"""R2Store.get retries a body stream that breaks mid-read, and raises client errors at once."""
import pytest
from botocore.exceptions import ClientError, ResponseStreamingError

from signtranslator.data.text_motion_dataset import R2Store


class Body:
    def __init__(self, fail):
        self.fail = fail
    def read(self):
        if self.fail:
            raise ResponseStreamingError(error=BrokenPipeError(32, "Broken pipe"))
        return b"payload"


class Client:
    def __init__(self, failures, missing=False):
        self.failures, self.missing, self.calls = failures, missing, 0
    def get_object(self, Bucket, Key):
        self.calls += 1
        if self.missing:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": Body(self.calls <= self.failures)}


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
