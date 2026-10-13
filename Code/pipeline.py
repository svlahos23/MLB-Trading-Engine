from __future__ import annotations

import ast
import json
import time
from typing import Any, cast

import httpx
import requests
from openai import OpenAI


def mlb_big_boy(user_statement: str, api_key: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Collect 30 MLB 40-man rosters, evaluate players, parse user preferences, return two verified pools."""
    BATCH_SIZE = 5
    MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
    ROSTER_TYPE = "40Man"
    REQUEST_DELAY = 0.15

    MLB_PROMPT_1 = """Below is a list of current MLB players grouped by team.

    {batch_text}

    Evaluate EVERY player listed.

    For each player, return:

    1. Attributes:

       - Up to 8 concise attributes describing the player physically,

         mentally, stylistically, or by career stage.

       - Examples include:
         age,
         young,
         veteran,
         speed,
         quickness,
         strength,
         size,
         athleticism,
         durability,
         temperament,
         baseball IQ,
         postseason experience,
         postseason riser,
         clutch,
         inconsistent,
         choker,
         high-motor,
         injury-prone,
         etc.

    2. Strengths:

       - Up to 5 meaningful CURRENT baseball strengths.

       - Possible areas include:
         hitting,
         contact hitting,
         power hitting,
         home run power,
         plate discipline,
         pitch recognition,
         bat-to-ball skills,
         on-base ability,
         hitting for average,
         hitting against left-handed pitching,
         hitting against right-handed pitching,
         situational hitting,
         baserunning,
         stolen bases,
         speed,
         fielding,
         infield defense,
         outfield defense,
         range,
         arm strength,
         throwing accuracy,
         defensive versatility,
         catching,
         pitch framing,
         game calling,
         pitching,
         fastball command,
         breaking balls,
         offspeed pitches,
         strikeout ability,
         control,
         command,
         limiting walks,
         inducing ground balls,
         run prevention,
         starting pitching,
         relief pitching,
         durability,
         baseball IQ,
         decision making,
         etc.

    3. Weaknesses:

       - Up to 5 meaningful CURRENT baseball weaknesses.

       - Use the same types of baseball categories as the strengths.

    IMPORTANT RULES:

    - Be stringent.

    - Evaluate CURRENT ability only.

    - Do NOT evaluate players based on what they were at their historical peak.

    - Do NOT evaluate them based on what they might become in the future.

    - Do NOT manufacture strengths or weaknesses simply to reach 5.

    - If a player only has 2 clear weaknesses, return 2.

    - If a player only has 3 clear strengths, return 3.

    - Use current information when necessary.

    - Distinguish between reputation and current performance.

    - Do not call someone clutch or a choker unless there is meaningful evidence.

    - Do not infer personality traits without reasonable public evidence.

    - Baseball skills should be concrete enough that another program can

      later use them to evaluate trade preferences.

    - Evaluate every player in the supplied batch.

    - Do not omit players.

    Return the information EXACTLY in this general format:

    Team Name:

    Player Name: Attributes ["attribute 1", "attribute 2", ...], Strengths ["strength 1", "strength 2", ...], Weaknesses ["weakness 1", "weakness 2", ...]

    Player Name 2: Attributes ["attribute 1", "attribute 2", ...], Strengths ["strength 1", "strength 2", ...], Weaknesses ["weakness 1", "weakness 2", ...]

    Next Team Name:

    Player Name: Attributes [...], Strengths [...], Weaknesses [...]

    Player Name 2: Attributes [...], Strengths [...], Weaknesses [...]

    Return ONLY the requested team/player information.

    Do not include:

    - an introduction

    - an explanation

    - methodology

    - citations

    - source lists

    - conclusions

    - disclaimers"""

    MLB_PROMPT_2 = """From this statement, you must parse multiple pieces of information.

    The team the user is working for, which players (or what type of baseball players) 

    they are looking for to join their team, what players (or what type of baseball players) they are willing to part with for this trade, 

    what players (or what type of baseball players) on their team are untouchable, what players you would like avoid acquiring, and what 

    weaknesses they want to avoid in acquired players. 

    This statement in particular {user_statement}

    Return the information in the following format:

    “Team Name: [Insert Team Name]

    Players Wanted: [Insert Player Names]

    Player Types Wanted: [Insert Player Types Wanted]

    Players Willing to Part With: [Insert Players Willing to Part With]

    Untouchable Players: [Insert Untouchable Players]

    Untouchable Player Types: [Insert Untouchable Player Types]

    Players to Avoid: [Insert Players to Avoid]

    Types of Players to Avoid: [Insert Types of Players to Avoid]

    Teams to Avoid: [Insert Teams to Avoid]”

    This is the only thing that should be returned. Do not include any other text or explanation."""

    MLB_PROMPT_3 = """You are reducing the search space for an MLB trade-generation engine.

    You are given:

    1. A user's natural-language trade demands.

    2. Formatted information for MLB players containing their team,

       Attributes, Strengths, and Weaknesses.

    Your job is to reduce the trade search space in TWO stages.

    ============================================================

    USER DEMANDS

    ============================================================

    {demands}

    ============================================================

    FORMATTED PLAYER INFORMATION

    ============================================================

    {all_results}

    ============================================================

    STAGE 1 — REMOVE DISALLOWED PLAYERS AND TEAMS

    ============================================================

    First determine which team the user represents from the USER DEMANDS.

    For the USER'S TEAM:

    - Remove any player explicitly identified as untouchable.

    - Remove any player who matches an untouchable player type, skill,

      attribute, strength, weakness, or characteristic described by the user.

    - Use the player's Attributes, Strengths, and Weaknesses to determine

      whether they match the user's restriction.

    - Interpret restrictions semantically rather than requiring exact

      keyword matches.

    Examples:

    If the user says they will not trade good hitters, protect players
    whose Strengths indicate things such as:

    - contact hitting
    - power hitting
    - hitting for average
    - on-base ability
    - plate discipline
    - pitch recognition
    - run production

    If the user says they will not trade good pitchers, protect players
    whose Strengths clearly indicate strong pitching value.

    The players remaining on the user's team represent players the user

    is willing to trade away.

    For EVERY OTHER TEAM:

    - Remove any player the user explicitly says they do not want.

    - Remove any player matching a player type, weakness, characteristic,

      attribute, or skill that the user says they want to avoid.

    - Use each player's Attributes, Strengths, and Weaknesses.

    - Interpret the user's wording semantically.

    - Do not require exact keyword matches.

    Examples:

    "bad hitter" may correspond to:

    - poor contact hitting
    - weak plate discipline
    - low on-base ability
    - high strikeout rate
    - poor pitch recognition
    - inconsistent offensive production

    "choker" should only match a player if their supplied player information
    actually indicates poor clutch performance, postseason decline, choking,
    or an equivalent characteristic.

    Do NOT infer a characteristic that is not reasonably supported by the

    provided player information.

    ============================================================

    TEAM RESTRICTIONS

    ============================================================

    If the user explicitly says they do not want to trade with a team:

    - Remove that ENTIRE team.

    - Do not include that team in Stage 1.

    - Do not include that team in Stage 2.

    A team-level restriction takes precedence over a request for an

    individual player on that team unless the user's demands clearly

    indicate an exception.

    ============================================================

    EXPLICITLY WANTED PLAYERS

    ============================================================

    If the user explicitly names a player they want to acquire:

    - Keep that player regardless of player-level avoidance filters.

    - Include that player in Stage 1 if their team is still eligible.

    - Include that player in Stage 2 if their team is still eligible.

    Explicitly wanted players override:

    - unwanted player-type filters

    - unwanted skill filters

    - unwanted weakness filters

    - attribute filters

    - general player avoidance preferences

    However, an explicit TEAM restriction takes precedence.

    ============================================================

    STAGE 1 OUTPUT

    ============================================================

    Stage 1 represents the complete remaining search space.

    For the user's team:

    Include only players who remain tradeable after applying the user's

    untouchable-player and untouchable-player-type restrictions.

    For opposing teams:

    Include only players who remain allowable after applying:

    - explicit player exclusions

    - undesirable player-type exclusions

    - undesirable skill or attribute exclusions

    - team exclusions

    Do not remove an opposing player simply because they do not match

    something the user wants.

    That matching happens in Stage 2.

    Stage 1 answers:

    "Which players are still allowed to participate in a possible trade?"

    ============================================================

    STAGE 2 — IDENTIFY ACTUAL TRADE TARGETS

    ============================================================

    Using ONLY opposing-team players that survived Stage 1, create a

    second and narrower search space.

    Do NOT include players from the user's own team in Stage 2.

    A player belongs in Stage 2 if ANY of these conditions are true:

    1. The user explicitly names that player as someone they want.

    OR

    2. The player meaningfully matches a player TYPE the user wants.

    OR

    3. The player meaningfully matches a skill, role, strength,

       characteristic, or attribute the user wants.

    Examples include:

    - backup catcher
    - starting catcher
    - contact hitter
    - power hitter
    - on-base specialist
    - starting pitcher
    - relief pitcher
    - closer
    - strikeout pitcher
    - ground-ball pitcher
    - good defensive infielder
    - good defensive outfielder
    - versatile defender
    - base stealer
    - athlete
    - young player
    - veteran
    - high-IQ player
    - postseason performer
    - clutch player

    Interpret these concepts semantically.

    For example:

    "strong backup catcher" could reasonably match catchers whose supplied
    information indicates several relevant traits such as:

    - pitch framing
    - game calling
    - blocking
    - throwing accuracy
    - arm strength
    - catching defense
    - contact hitting
    - durability

    The player does NOT need to literally have the phrase
    "strong backup catcher" in their profile.

    However:

    - Do not stretch weak similarities.

    - Do not include someone merely because they vaguely resemble the request.

    - There should be meaningful evidence in their supplied Attributes,

      Strengths, or Weaknesses.

    Explicitly wanted players must appear in Stage 2 if their team remains

    eligible.

    Stage 2 answers:

    "Which of the allowable opposing players actually match what the user

    wants to acquire?"

    ============================================================

    IMPORTANT LOGIC RULES

    ============================================================

    Apply the filters in this order:

    1. Determine the user's team.

    2. Identify explicitly excluded teams.

    3. Identify explicitly untouchable players on the user's team.

    4. Identify untouchable player TYPES on the user's team.

    5. Identify explicitly unwanted opposing players.

    6. Identify unwanted opposing player TYPES.

    7. Apply the mandatory exception for explicitly wanted players,

       unless their team is prohibited.

    8. Construct Stage 1.

    9. From Stage 1 opposing players only, identify players matching

       the user's desired players and desired player types.

    10. Construct Stage 2.

    Do not invent information.

    Use ONLY:

    - the user's demands

    - the supplied player Attributes

    - the supplied player Strengths

    - the supplied player Weaknesses

    - the supplied team/player associations

    Do not use outside knowledge.

    Do not perform web searches.

    Do not move a player to a different team.

    Preserve all player names exactly as supplied.

    Preserve all team names exactly as supplied.

    Do not include duplicate players.

    Do not include teams with zero remaining players.

    ============================================================

    OUTPUT FORMAT

    ============================================================

    Return EXACTLY TWO valid Python dictionaries in raw text.

    The FIRST dictionary is Stage 1.

    The SECOND dictionary is Stage 2.

    Do NOT include:

    - headings

    - labels

    - explanations

    - reasoning

    - comments

    - Markdown

    - Markdown code fences

    - variable assignments

    - introductory text

    - concluding text

    - any text between the two dictionaries

    Do NOT return:

    stage_1 = ...

    Do NOT return:

    stage_2 = ...

    Return this structure:

    {

        "Team Name": ["Player Name", "Player Name"],

        "Another Team": ["Player Name", "Player Name"]

    }

    {

        "Team Name": ["Player Name"],

        "Another Team": ["Player Name"]

    }

    The response must contain exactly two Python dictionary expressions

    and nothing else."""

    def collect_mlb_rosters() -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
        session = requests.Session()
        session.headers.update({"User-Agent": "MLBTradeEngine/1.0", "Accept": "application/json"})

        def fetch(path: str, params: dict | None = None) -> dict:
            for attempt in range(3):
                try:
                    response = session.get(f"{MLB_STATS_API}{path}", params=params, timeout=25)
                    response.raise_for_status()
                    return response.json()
                except (requests.RequestException, ValueError) as exc:
                    if attempt == 2: raise RuntimeError(f"MLB Stats API request failed: {path}") from exc
                    time.sleep(attempt + 1)
            raise RuntimeError("Unreachable request state")

        teams = fetch("/teams", {"sportId": 1, "activeStatus": "Y"}).get("teams", [])
        teams = sorted((team for team in teams if team.get("sport", {}).get("id") == 1), key=lambda team: team["name"])
        if len(teams) != 30: raise RuntimeError(f"Expected 30 MLB teams; received {len(teams)}")

        league_players: dict[str, list[str]] = {}
        player_ids: dict[str, dict[str, int]] = {}
        for index, team in enumerate(teams, 1):
            name = team["name"]
            roster = fetch(f"/teams/{team['id']}/roster", {"rosterType": ROSTER_TYPE}).get("roster", [])
            players = {person["fullName"]: person["id"] for row in roster if (person := row.get("person", {})).get("id") and person.get("fullName")}
            if not players: raise RuntimeError(f"Empty MLB roster for {name}; stopping rather than evaluating incomplete data")
            league_players[name] = sorted(players)
            player_ids[name] = players
            print(f"[{index:02d}/30] {name}: {len(players)} players")
            time.sleep(REQUEST_DELAY)
        return league_players, player_ids

    if not isinstance(user_statement, str) or not user_statement.strip(): raise ValueError("user_statement must be a nonempty string")
    if not isinstance(api_key, str) or not api_key.strip(): raise ValueError("api_key must be a nonempty string")
    league_players, player_ids = collect_mlb_rosters()
    client = OpenAI(api_key=api_key, http_client=cast(Any, httpx.Client()))
    teams = list(league_players.items())
    all_results: list[str] = []
    total_batches = (len(teams) + BATCH_SIZE - 1) // BATCH_SIZE

    # STAGE 2 — Evaluate players with the user's original MLB player-profile prompt.
    for start_index in range(0, len(teams), BATCH_SIZE):
        batch = dict(teams[start_index:start_index + BATCH_SIZE])
        batch_text = json.dumps(batch, ensure_ascii=False, indent=2)
        print(f"MLB player evaluation batch {(start_index // BATCH_SIZE) + 1}/{total_batches}")
        response = client.responses.create(model="gpt-5.6", tools=[{"type": "web_search"}], input=MLB_PROMPT_1.replace("{batch_text}", batch_text))
        all_results.append(response.output_text.strip())

    # STAGE 3 — Interpret user demands using the original preference prompt.
    demands_response = client.responses.create(model="gpt-5.6", input=MLB_PROMPT_2.replace("{user_statement}", user_statement))
    demands = demands_response.output_text.strip()
    print("\nPARSED MLB USER PREFERENCES:\n", demands)

    # STAGE 4 — Reduce search space using the original filtering prompt.
    filtered_response = client.responses.create(model="gpt-5.6", input=MLB_PROMPT_3.replace("{demands}", demands).replace("{all_results}", "\n\n".join(all_results)))
    raw_output = filtered_response.output_text.strip()
    try:
        tree = ast.parse(raw_output, mode="exec")
        dictionaries = []
        for node in tree.body:
            if not isinstance(node, ast.Expr): raise ValueError("Output must contain only dictionary expressions")
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict): raise ValueError("Expected a dictionary expression")
            dictionaries.append(value)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid MLB filtering output:\n{raw_output}") from exc
    if len(dictionaries) != 2: raise ValueError(f"Expected two player dictionaries:\n{raw_output}")

    stage_1_players, stage_2_targets = dictionaries
    for label, pool in (("Stage 1", stage_1_players), ("Stage 2", stage_2_targets)):
        for team, names in pool.items():
            if team not in player_ids: raise ValueError(f"{label}: GPT returned an unknown team: {team}")
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names): raise ValueError(f"{label}: Invalid players for {team}")
            if len(names) != len(set(names)): raise ValueError(f"{label}: Duplicate player in {team}")
            invalid = [name for name in names if name not in player_ids[team]]
            if invalid: raise ValueError(f"{label}: Names not on the scraped {team} roster: {invalid}")
    for team, targets in stage_2_targets.items():
        if any(name not in stage_1_players.get(team, []) for name in targets): raise ValueError(f"Stage 2 includes a player excluded from Stage 1: {team}")

    print("\nSTAGE 1 — ALLOWABLE MLB TRADE PLAYERS:\n", stage_1_players)
    print("\nSTAGE 2 — MATCHING MLB TRADE TARGETS:\n", stage_2_targets)
    return stage_1_players, stage_2_targets
