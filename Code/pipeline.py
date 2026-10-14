from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Callable, cast

import httpx
import requests
from openai import OpenAI


def mlb_big_boy(user_statement: str, api_key: str) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]],
    dict[str, list[str]], list[Callable[[set[str], set[str]], bool]]
]:
    """Collect MLB 40-man rosters and return four preference-filtered pools
    plus callable, explicit AND/OR/NOT trade requirements. No trade generation
    or MLB contract/legal eligibility validation is performed.
    """
    BATCH_SIZE = 5
    MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
    ROSTER_TYPE = "40Man"
    REQUEST_DELAY = 0.15

    if not isinstance(user_statement, str) or not user_statement.strip():
        raise ValueError("user_statement must be a nonempty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a nonempty string")

    def normalize_name(value):
        return re.sub(r"\s+", " ", str(value)).strip()

    # ============================================================
    # STAGE 1 — COLLECT MLB 40-MAN ROSTERS
    # ============================================================
    
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
                    if attempt == 2:
                        raise RuntimeError(f"MLB Stats API request failed: {path}") from exc
                    time.sleep(attempt + 1)
            raise RuntimeError("Unreachable request state")

        teams = fetch("/teams", {"sportId": 1, "activeStatus": "Y"}).get("teams", [])
        teams = sorted((team for team in teams if team.get("sport", {}).get("id") == 1), key=lambda team: team["name"])
        if len(teams) != 30:
            raise RuntimeError(f"Expected 30 MLB teams; received {len(teams)}")

        league_players: dict[str, list[str]] = {}
        player_ids: dict[str, dict[str, int]] = {}
        for index, team in enumerate(teams, 1):
            name = team["name"]
            roster = fetch(f"/teams/{team['id']}/roster", {"rosterType": ROSTER_TYPE}).get("roster", [])
            players = {person["fullName"]: person["id"] for row in roster
                       if (person := row.get("person", {})).get("id") and person.get("fullName")}
            if not players:
                raise RuntimeError(f"Empty MLB roster for {name}; stopping rather than evaluating incomplete data")
            league_players[name] = sorted(players)
            player_ids[name] = players
            print(f"[{index:02d}/30] {name}: {len(players)} players")
            time.sleep(REQUEST_DELAY)
        return league_players, player_ids

    league_players, player_ids = collect_mlb_rosters()
    client = OpenAI(api_key=api_key, http_client=cast(Any, httpx.Client()))

    # ============================================================
    # STAGE 2 — CURRENT PLAYER EVALUATIONS (5 TEAMS PER GPT BATCH)
    # ============================================================
    
    MLB_PROMPT_1 = """Below is a list of current MLB players grouped by team.

{batch_text}

Evaluate EVERY player listed.
For each player, return:

1. Attributes: Up to 8 concise CURRENT attributes describing players physically,
mentally, stylistically, or by career stage. Examples: age, young, veteran,
speed, quickness, strength, size, athleticism, durability, temperament,
baseball IQ, postseason experience, postseason riser, clutch, inconsistent,
choker, high-motor, injury-prone.

2. Strengths: Up to 5 meaningful CURRENT baseball strengths, where supported.
Possible areas: hitting, contact hitting, power hitting, home run power,
plate discipline, pitch recognition, bat-to-ball skills, on-base ability,
hitting for average, splits against left/right-handed pitching,
situational hitting, baserunning, stolen bases, speed, fielding, infield
and outfield defense, range, arm strength, throwing accuracy, defensive
versatility, catching, pitch framing, game calling, pitching, fastball
command, breaking balls, offspeed pitches, strikeout ability, control,
command, limiting walks, inducing ground balls, run prevention, starting
pitching, relief pitching, durability, baseball IQ, decision making.

3. Weaknesses: Up to 5 meaningful CURRENT baseball weaknesses, drawing from
the same categories as strengths.

IMPORTANT RULES:
- Be stringent. Evaluate CURRENT ability only; not historical peak or potential.
- Do NOT manufacture strengths or weaknesses to reach the maximum count.
- Distinguish reputation from current performance.
- Only label someone clutch or a choker with meaningful evidence.
- Do not infer personality traits without reasonable public evidence.
- Be concrete enough to support later trade filtering.
- Evaluate every player, without omissions. Use current information when necessary.

Return exactly the following format, for every team and player:
Team Name:
Player Name: Attributes ["attribute 1", ...], Strengths ["strength 1", ...], Weaknesses ["weakness 1", ...]
Player Name 2: Attributes [...], Strengths [...], Weaknesses [...]
Next Team Name:
Player Name: Attributes [...], Strengths [...], Weaknesses [...]

Only requested team/player information. No introductions, methodology,
citations, source lists, conclusions or disclaimers.
"""

    teams = list(league_players.items())
    all_results: list[str] = []
    total_batches = (len(teams) + BATCH_SIZE - 1) // BATCH_SIZE
    for start_index in range(0, len(teams), BATCH_SIZE):
        batch_text = json.dumps(dict(teams[start_index:start_index + BATCH_SIZE]), ensure_ascii=False, indent=2)
        print(f"MLB player evaluation batch {(start_index // BATCH_SIZE) + 1}/{total_batches}")
        response = client.responses.create(
            model="gpt-5.6", tools=[{"type": "web_search"}],
            input=MLB_PROMPT_1.replace("{batch_text}", batch_text)
        )
        all_results.append(response.output_text.strip())

    # ============================================================
    # STAGE 3 — PARSE THE USER'S TRADE PREFERENCES USING GPT
    # ============================================================
    
    demands = client.responses.create(
        model="gpt-5.6",
        input=f"""
You are interpreting an MLB trade request for a later player-pool filter.
Read the user statement and extract the following, without inventing names:

Team Name: [user's MLB team]
Required Acquisition Players: [specifically named players the user insists on acquiring]
Preferred Acquisition Players: [named players the user is interested in but does not require]
Acquisition Player Types: [roles, skills, traits of other acceptable acquisitions]
Preferred Outgoing Players: [named players the user explicitly wants to trade away or offers to trade]
Other Allowed Outgoing Players: [named players the user explicitly permits trading, without preferring to move]
Untouchable Players: [named players never to trade]
Untouchable Player Types: [traits of players never to trade]
Players to Avoid Acquiring: [specific names]
Player Types to Avoid Acquiring: [traits, weaknesses, roles to exclude]
Teams to Avoid: [teams the user refuses to deal with]
Required Trade Logic: [Python Boolean membership expressions, as quoted strings]

Extract mandatory logical conditions ONLY when explicitly stated by the user.
Use Python expressions with membership tests, AND/OR/NOT operators, parentheses,
and the variables acquired and outgoing. Use full player names, for example:
Required Trade Logic: ["('Player A' in acquired or 'Player B' in acquired)", "not ('Player A' in acquired and 'Player B' in acquired)", "('Player C' not in outgoing or 'Player A' in acquired)"]
Treat "either X or Y, but not both" as TWO expressions: the OR and the NOT-AND.
For "trade X only if receiving Y", use "X not in outgoing or Y in acquired".
If the user requires both X and Y, use one AND expression (or equivalent
individual membership expressions).
Use acquired for players the user's team RECEIVES and outgoing for players
it SENDS. Avoid assumptions about other teams' player flows.
Do not invent constraints from preferred or merely allowable players.
Do not use player-type conditions, function calls, Python lambdas, or variables
besides acquired/outgoing. If no explicit mandatory condition exists, use [].

Distinguish REQUIRED from merely PREFERRED acquisitions. Explicitly naming
someone does not necessarily make acquisition mandatory: follow the wording.
Likewise, "willing to part with" means allowed, not necessarily preferred,
unless the user expresses a desire to move that player.
If a category is absent, put [] rather than speculating.

USER STATEMENT:
{user_statement}

Return only the labeled fields above. No explanations or Markdown.
"""
    )

    print("\nPARSED USER PREFERENCES:\n", demands.output_text)
    # Parse the required trade-logic list separately from the preference labels.
    logic_match = re.search(r"(?im)^Required Trade Logic:\s*(\[[^\n]*\])\s*$", demands.output_text)
    if logic_match is None:
        raise ValueError("GPT omitted Required Trade Logic list")
    raw_logic = ast.literal_eval(logic_match.group(1))
    if not isinstance(raw_logic, list) or not all(isinstance(x, str) for x in raw_logic):
        raise ValueError("Required Trade Logic must be a list of Python expressions")

    # Accept only name membership and AND/OR/NOT. Do not execute GPT-generated code.
    # These predicates are evaluated later, once for each proposed trade.
    def make_trade_rule(expression):
        tree = ast.parse(expression, mode="eval")

        def check_syntax(node):
            if isinstance(node, ast.Expression):
                return check_syntax(node.body)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                if len(node.values) < 2:
                    raise ValueError("Boolean group requires at least two members")
                for child in node.values:
                    check_syntax(child)
                return
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return check_syntax(node.operand)
            if (isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and isinstance(node.comparators[0], ast.Name)
                    and node.comparators[0].id in {"acquired", "outgoing"}):
                return
            raise ValueError(f"Unsupported trade logic expression: {expression!r}")

        check_syntax(tree)

        def evaluate(node, acquired, outgoing):
            if isinstance(node, ast.Expression):
                return evaluate(node.body, acquired, outgoing)
            if isinstance(node, ast.BoolOp):
                results = (evaluate(n, acquired, outgoing) for n in node.values)
                return all(results) if isinstance(node.op, ast.And) else any(results)
            if isinstance(node, ast.UnaryOp):
                return not evaluate(node.operand, acquired, outgoing)
            selected = acquired if node.comparators[0].id == "acquired" else outgoing
            present = node.left.value in selected
            return present if isinstance(node.ops[0], ast.In) else not present

        def rule(acquired, outgoing):
            return evaluate(tree, set(acquired), set(outgoing))

        rule.expression = expression
        return rule

    # Verify all referenced players against the current MLB rosters. Keep the
    # resulting predicates intact even when an eligibility/preference filter
    # excludes a named player: missing that player will cause the relevant
    # requirement to fail when checking an actual trade.
    known_names = {normalize_name(p).casefold(): p for roster in league_players.values() for p in roster}
    canonical_logic = []
    for expression in raw_logic:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                original = node.left.value
                actual = known_names.get(normalize_name(original).casefold())
                if actual is None:
                    raise ValueError(f"Required trade logic references player not on a scraped roster: {original!r}")
                node.left.value = actual
        canonical_logic.append(ast.unparse(tree))
    required_logic = [make_trade_rule(expression) for expression in canonical_logic]


    # ============================================================
    # STAGE 4 — GPT PLAYER-POOL REDUCTION AND PRIORITIZATION
    # Separate desired/required acquisitions from permitted/preferred exits.
    # No CBA evaluation is performed in this GPT stage.
    # ============================================================
    
    response = client.responses.create(
        model="gpt-5.6",
        input=f"""
You build four DIFFERENT player pools for an MLB trade search.

USER PREFERENCES:
{demands.output_text}

PLAYER PROFILES AND THEIR CURRENT TEAMS:
{all_results}

Using ONLY the supplied team rosters and player attributes/strengths/weaknesses,
produce EXACTLY FOUR Python dictionaries in the order specified below.
Each dictionary maps exact existing MLB team names to lists of exact existing
player names. No duplicates, fabricated names, or empty teams.

DICTIONARY 1: required_targets
- On OTHER teams, players the user MUST acquire, per explicit mandatory wording.
- Do NOT classify all named players as required automatically: distinguish an
  actual condition from a mere preference or interest.
- Honor excluded teams even if the user names a player on such a team, unless
  the user explicitly grants an exception for that team.
- A specifically required acquisition overrides general skill/weakness dislikes,
  but not an explicit team ban.

DICTIONARY 2: tradeable_targets
- On OTHER teams, players the user is interested in acquiring based on requested
  player types, skills, roles or explicit nonmandatory player interests.
- Only include players who reasonably match, using supplied scouting profiles.
- Respect excluded teams, explicitly unwanted players and unwanted player types.
- Do NOT duplicate players already in required_targets.
- Do not include everyone generally allowable: these must be actual targets.

DICTIONARY 3: wanted_to_trade_away
- From the USER'S OWN team ONLY, players the user expresses a genuine desire
  to trade away (not merely willingness to part with).
- Never include untouchables unless the user explicitly overrides that status.

DICTIONARY 4: can_trade_away
- From the USER'S OWN team ONLY, all other players the user is willing to trade.
- Exclude explicitly untouchable names and players matching protected types.
- Preserve explicit user permission to move a player even if they otherwise
  match a broad protected type (e.g. willing to part with Player D but
  protecting other good hitters).
- Include named players the user is willing to part with, even if not eager.
- Include all other nonprotected roster players unless the request clearly
  says ONLY specific players may be traded.
- Do NOT duplicate any player already in wanted_to_trade_away.

These four dictionaries must be pairwise DISJOINT. Put every explicitly required
OR-alternative into required_targets, even if no single member is mandatory;
the Required Trade Logic field controls acceptable combinations.
They describe USER
preferences only, NOT legal trade eligibility. Do not apply salary/CBA rules.
Do NOT use external knowledge or web searches. Do NOT change a player's team.

OUTPUT:
Return exactly four raw Python dict expressions (not a list or variable
assignments), in this strict order:
1. required_targets
2. tradeable_targets
3. wanted_to_trade_away
4. can_trade_away

Example output structure (illustrative names only):
{{"Opposing Team": ["Required Player"]}}
{{"Other Team": ["Interesting Player"]}}
{{"My Team": ["Player To Move"]}}
{{"My Team": ["Other Allowed Player"]}}

No headings, explanation, code fences, or text between dicts.
"""
    )

    def parse_four_dictionaries(text):
        try:
            tree = ast.parse(text.strip(), mode="exec")
        except SyntaxError as exc:
            raise ValueError(f"GPT returned invalid pool syntax:\n{text}") from exc

        values = []
        for node in tree.body:
            if not isinstance(node, ast.Expr):
                raise ValueError(f"Unexpected non-dictionary expression:\n{text}")
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError("Expected exactly four dictionaries")
            values.append(value)
        if len(values) != 4:
            raise ValueError(f"Expected four dictionaries, got {len(values)}:\n{text}")
        return values

    required_targets, tradeable_targets, wanted_to_trade_away, can_trade_away = (
        parse_four_dictionaries(response.output_text)
    )

    # Validate GPT's names against the scraped roster: models can misspell or
    # assign players to the wrong team. Keep only verified membership.
    roster_lookup = {
        team: {normalize_name(name).casefold(): name for name in names}
        for team, names in league_players.items()
    }

    def verify_roster_pool(pool, pool_name):
        clean = {}
        for team, names in pool.items():
            if team not in roster_lookup:
                raise ValueError(f"{pool_name}: unknown team {team!r}")
            if not isinstance(names, list):
                raise ValueError(f"{pool_name}: player values must be lists")
            verified = []
            for name in names:
                actual = roster_lookup[team].get(normalize_name(name).casefold())
                if actual is None:
                    raise ValueError(f"{pool_name}: {name!r} not found on {team}")
                verified.append(actual)
            if verified:
                clean[team] = sorted(set(verified))
        return clean

    required_targets = verify_roster_pool(required_targets, "required_targets")
    tradeable_targets = verify_roster_pool(tradeable_targets, "tradeable_targets")
    wanted_to_trade_away = verify_roster_pool(wanted_to_trade_away, "wanted_to_trade_away")
    can_trade_away = verify_roster_pool(can_trade_away, "can_trade_away")

    # Favor the higher-priority group if GPT accidentally duplicates a player.
    seen = set()
    def deduplicate_pool(pool):
        cleaned = {}
        for team, players in pool.items():
            unique = [name for name in players if (team, name) not in seen]
            seen.update((team, name) for name in unique)
            if unique:
                cleaned[team] = unique
        return cleaned

    required_targets = deduplicate_pool(required_targets)
    tradeable_targets = deduplicate_pool(tradeable_targets)
    wanted_to_trade_away = deduplicate_pool(wanted_to_trade_away)
    can_trade_away = deduplicate_pool(can_trade_away)

    print("\nREQUIRED TARGETS (BEFORE CBA):", required_targets)
    print("\nTRADEABLE TARGETS (BEFORE CBA):", tradeable_targets)
    print("\nWANTED TO TRADE AWAY (BEFORE CBA):", wanted_to_trade_away)
    print("\nCAN TRADE AWAY (BEFORE CBA):", can_trade_away)

    # Check that mandatory names can be sourced from the appropriate pools.
    # This is an early feasibility diagnostic; no MLB transaction legality is claimed.
    all_incoming = {p for pool in (required_targets, tradeable_targets) for players in pool.values() for p in players}
    all_outgoing = {p for pool in (wanted_to_trade_away, can_trade_away) for players in pool.values() for p in players}
    for expression in canonical_logic:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                name = node.left.value
                side = node.comparators[0].id
                # Negative membership may intentionally refer to a player that
                # the existing filters already prevent from moving.
                if isinstance(node.ops[0], ast.In):
                    if side == "acquired" and name not in all_incoming:
                        print(f"WARNING: logic references acquisition outside filtered target pools: {name}")
                    if side == "outgoing" and name not in all_outgoing:
                        print(f"WARNING: logic references outgoing player outside filtered pools: {name}")

    print("\nFINAL MLB PLAYER DICTIONARIES — AFTER USER-PREFERENCE FILTERING")
    print("\nREQUIRED TARGETS:", required_targets)
    print("\nTRADEABLE TARGETS:", tradeable_targets)
    print("\nWANTED TO TRADE AWAY:", wanted_to_trade_away)
    print("\nCAN TRADE AWAY:", can_trade_away)
    print("\nREQUIRED TRADE LOGIC:", [rule.expression for rule in required_logic])
    return required_targets, tradeable_targets, wanted_to_trade_away, can_trade_away, required_logic
