from enum import Enum
from typing import List, Literal, Union

from pydantic import BaseModel, Field


# --- Customer Support Schemas ---


class SupportIntent(str, Enum):
    REFUND = "Refund Request"
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING = "Billing Inquiry"
    PRODUCT_INFO = "Product Information"
    WARRANTY = "Warranty Claim"
    COMPLAINT = "General Complaint"
    OTHER = "Other"


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ActionType(str, Enum):
    CLOSE = "Close"
    ESCALATION = "Escalation"
    KEEP_OPEN = "Keep Open"


class ToolCall(str, Enum):
    KNOWLEDGE_BASE_SEARCH = "Knowledge Base Search"
    NONE = "none"


class CustomerSupportInteraction(BaseModel):
    agent_reasoning: str = Field(
        description="Step-by-step reasoning to reach the final response. Explain the diagnosis and decision process."
    )
    intent: str = Field(description="Primary intent of the customer")
    urgency: str = Field(description="Assessed urgency level")
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (Very Negative) to 1.0 (Very Positive)"
    )
    key_entities: List[str] = Field(
        description="Key entities extracted (Product names, Order IDs, Dates)"
    )
    missing_information: List[str] = Field(
        description="Information missing to resolve the query"
    )
    action: ActionType = Field(
        description="The action taken by the agent. Close if it's resolved, escalade if it's urgent, and keep it open if it's pending customer."
    )
    action_reason: str = Field(description="Reason for the action taken by the agent.")
    query: str = Field(
        description="The search query that you would need to run against the knowledge base to resolve the customer request."
    )
    response: str = Field(
        description="The final natural language response to the customer"
    )


# --- Tool Calling Schemas ---

# Type aliases for reusability
RoomName = Literal["kitchen", "bedroom", "living_room", "kids_room", "bathroom"]
DoorName = Literal["front_door", "back_door", "garage"]
ThermostatMode = Literal["cool", "heat", "auto"]


class TurnOnLightArgs(BaseModel):
    room: RoomName = Field(description="The name of the room.")
    brightness: int = Field(80, description="Brightness percentage (0-100).")
    color: str = Field("white", description="Color of the light.")


class TurnOnLight(BaseModel):
    """Turn on a light in a specific room."""

    name: Literal["turn_on_light"] = "turn_on_light"
    arguments: TurnOnLightArgs


class TurnOffLightArgs(BaseModel):
    room: RoomName = Field(description="The name of the room.")


class TurnOffLight(BaseModel):
    """Turn off a light in a specific room."""

    name: Literal["turn_off_light"] = "turn_off_light"
    arguments: TurnOffLightArgs


class SetThermostatArgs(BaseModel):
    temperature: float = Field(description="Target temperature in Celsius.")
    mode: ThermostatMode = Field("auto", description="Thermostat mode.")


class SetThermostat(BaseModel):
    """Set the thermostat temperature and mode."""

    name: Literal["set_thermostat"] = "set_thermostat"
    arguments: SetThermostatArgs


class PlayMusicArgs(BaseModel):
    genre: str = Field(description="Music genre.")
    volume: int = Field(50, description="Volume level (0-100).")


class PlayMusic(BaseModel):
    """Play music in a specific genre."""

    name: Literal["play_music"] = "play_music"
    arguments: PlayMusicArgs


class LockDoorArgs(BaseModel):
    door: DoorName = Field(description="Which door to lock.")


class LockDoor(BaseModel):
    """Lock a specific door."""

    name: Literal["lock_door"] = "lock_door"
    arguments: LockDoorArgs


class CheckWeatherArgs(BaseModel):
    location: str = Field(description="City name.")


class CheckWeather(BaseModel):
    """Check the weather for a specific location."""

    name: Literal["check_weather"] = "check_weather"
    arguments: CheckWeatherArgs


# Available tools list - single source of truth
AVAILABLE_TOOLS = (
    TurnOnLight,
    TurnOffLight,
    SetThermostat,
    PlayMusic,
    LockDoor,
    CheckWeather,
)


# Define the Union of all possible tool calls
class AnyToolCall(BaseModel):
    function: Union[
        TurnOnLight, TurnOffLight, SetThermostat, PlayMusic, LockDoor, CheckWeather
    ]


class ToolInvocation(BaseModel):
    reasoning: str = Field(
        description="Chain-of-thought reasoning for selecting the specific tool(s) and arguments."
    )
    response: str = Field(
        description="The final response to the user in natural language."
    )
    # We use a list to support multiple actions in one go
    tool_calls: List[AnyToolCall] = Field(
        description="A list of tool calls to execute."
    )
