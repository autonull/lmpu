"""Property tests for the protocol codec and model-plane parsing."""

from __future__ import annotations

import struct

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lmpu.protocol.codec import Frame, RPCCodec
from lmpu.protocol.messages import (
    RPCNotification,
    RPCRequest,
    RPCResponse,
    SubmitMessage,
    parse_model_message,
)

pytestmark = pytest.mark.property

JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=24),
)
JSON_VALUE = st.recursive(
    JSON_SCALAR,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=12), children, max_size=3),
    ),
    max_leaves=6,
)
JSON_OBJECT = st.dictionaries(st.text(min_size=1, max_size=16), JSON_VALUE, max_size=4)
METHOD = st.from_regex(r"lmpu\.[a-z][a-z0-9_.]{0,20}", fullmatch=True)
REQUEST_ID = st.one_of(st.integers(min_value=0, max_value=2**31), st.text(min_size=1, max_size=24))
RESPONSE_ID = st.one_of(st.none(), REQUEST_ID)


@settings(max_examples=50, deadline=None)
@given(request_id=REQUEST_ID, method=METHOD, params=JSON_OBJECT)
def test_rpc_request_frame_round_trip(
    request_id: int | str | None,
    method: str,
    params: dict[str, object],
) -> None:
    request = RPCRequest(id=request_id, method=method, params=params)
    encoded = RPCCodec.encode_request(request)

    length = struct.unpack(">I", encoded[:4])[0]
    assert length == len(encoded) - 4

    decoded = RPCCodec.decode_frame(Frame.decode(encoded)[0])
    assert isinstance(decoded, RPCRequest)
    assert decoded == request


@settings(max_examples=50, deadline=None)
@given(request_id=RESPONSE_ID, result=JSON_VALUE)
def test_rpc_response_frame_round_trip(
    request_id: int | str | None,
    result: object,
) -> None:
    response = RPCResponse(id=request_id, result=result)
    decoded = RPCCodec.decode_frame(Frame.decode(RPCCodec.encode_response(response))[0])

    assert isinstance(decoded, RPCResponse)
    assert decoded == response


@settings(max_examples=25, deadline=None)
@given(method=METHOD, params=JSON_OBJECT)
def test_rpc_notification_frame_round_trip(
    method: str,
    params: dict[str, object],
) -> None:
    notification = RPCNotification(method=method, params=params)
    encoded = RPCCodec.encode_notification(notification)
    decoded = RPCCodec.decode_frame(Frame.decode(encoded)[0])

    assert isinstance(decoded, RPCNotification)
    assert decoded == notification


@settings(max_examples=50, deadline=None)
@given(
    job_id=st.one_of(st.none(), st.text(min_size=1, max_size=24)),
    processor=st.text(min_size=1, max_size=24),
    op=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
    inputs=JSON_OBJECT,
)
def test_submit_message_round_trip(
    job_id: str | None,
    processor: str,
    op: str,
    inputs: dict[str, object],
) -> None:
    message = SubmitMessage(job_id=job_id, processor=processor, op=op, inputs=inputs)
    parsed = parse_model_message(message.model_dump(mode="json"))

    assert isinstance(parsed, SubmitMessage)
    assert parsed.job_id == message.job_id
    assert parsed.processor == message.processor
    assert parsed.op == message.op
    assert parsed.inputs == message.inputs
