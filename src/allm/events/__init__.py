"""Domain events: the platform's live feed (Roadmap M51).

The core already records *why* every belief changed in the append-only
store; the event log turns the changes the platform cares about — new
proposals, resolutions, confidence shifts — into an ordered, pollable
stream a frontend can subscribe to. It writes to the same versioned
store, so the feed is not a side-channel that can drift: it *is* data.
"""

from allm.events.log import Event, EventLog
from allm.events.webhooks import (
    ApprovalError,
    UrllibSender,
    WebhookDelivery,
    WebhookDispatcher,
    WebhookRegistry,
    WebhookSender,
    WebhookSubscription,
)

__all__ = [
    "ApprovalError",
    "Event",
    "EventLog",
    "UrllibSender",
    "WebhookDelivery",
    "WebhookDispatcher",
    "WebhookRegistry",
    "WebhookSender",
    "WebhookSubscription",
]
