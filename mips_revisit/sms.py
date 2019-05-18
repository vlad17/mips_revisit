"""
Send messages with the twilio api

This expects twilio creds in your env
"""

import os

from twilio.rest import Client


def get_twilio_creds_and_phone():
    """
    Return the twilio SID, token from env

    Expects the following to be environment vars,
    raises KeyError if they're not present.

    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TARGET_PHONE_FOR_SMS
    ORIGIN_PHONE_FOR_SMS
    """
    sidname = "TWILIO_ACCOUNT_SID"
    tokname = "TWILIO_AUTH_TOKEN"
    phone = "TARGET_PHONE_FOR_SMS"
    origin = "ORIGIN_PHONE_FOR_SMS"

    for x in [sidname, tokname, phone, origin]:
        if x not in os.environ:
            raise KeyError(x)

    account_sid = os.environ[sidname]
    auth_token = os.environ[tokname]
    phone = os.environ[phone]
    origin = os.environ[origin]

    return account_sid, auth_token, phone, origin


def makesms(body):
    """
    Best-effort send a message with environment-set
    twilio api and dst phone.

    Logs if twilio misconfigured
    """

    try:
        sid, token, phone, origin = get_twilio_creds_and_phone()
    except KeyError as e:
        from . import log

        log.debug(
            "error ({}) sending twilio message ({}), suppressing",
            e.args[0],
            body,
        )
        return

    client = Client(sid, token)

    client.messages.create(body=body, from_=origin, to=phone)
