import csv
import json
from pathlib import Path


REPLAY_PATH = Path("data/raw/sample/94733726.json")
OUTPUT_PATH = Path("data/processed/game_states.csv")

# Change this to the date of the Kaggle daily dataset
# that you downloaded this replay from.
SOURCE_DATE = "2026-08-18"


def get_player_features(player):
    active = [
        pokemon
        for pokemon in player.get("active", [])
        if isinstance(pokemon, dict)
    ]

    bench = [
        pokemon
        for pokemon in player.get("bench", [])
        if isinstance(pokemon, dict)
    ]

    pokemon_in_play = active + bench

    active_hp = active[0].get("hp", 0) if active else 0

    total_hp = sum(
        pokemon.get("hp", 0)
        for pokemon in pokemon_in_play
    )

    total_energy = sum(
        len(pokemon.get("energyCards", []))
        for pokemon in pokemon_in_play
    )

    active_has_status = int(
        player.get("asleep", False)
        or player.get("burned", False)
        or player.get("confused", False)
        or player.get("paralyzed", False)
        or player.get("poisoned", False)
    )

    return {
        "prizes_remaining": len(player.get("prize", [])),
        "hand_size": player.get("handCount", 0),
        "deck_size": player.get("deckCount", 0),
        "pokemon_in_play": len(pokemon_in_play),
        "active_hp": active_hp,
        "total_hp": total_hp,
        "total_energy": total_energy,
        "active_has_status": active_has_status,
    }


def get_winner(replay):
    rewards = replay.get("rewards", [])

    if rewards == [1, -1]:
        return 1

    if rewards == [-1, 1]:
        return 0

    # Draws or unusual/incomplete games are not usable
    # for our binary classification problem.
    return None


def process_replay(replay):
    p1_wins = get_winner(replay)

    if p1_wins is None:
        return []

    game_id = replay.get("info", {}).get(
        "EpisodeId",
        replay.get("id")
    )

    visualizer = replay["steps"][0][0].get("visualize")

    if not isinstance(visualizer, list):
        return []

    rows = []
    seen_turns = set()

    for frame in visualizer:
        current = frame.get("current")

        if not isinstance(current, dict):
            continue

        turn = current.get("turn", 0)

        # Skip setup states before the actual game starts.
        if turn < 1:
            continue

        # We only want the first state from each game turn.
        if turn in seen_turns:
            continue

        players = current.get("players", [])

        if len(players) != 2:
            continue

        seen_turns.add(turn)

        p1 = get_player_features(players[0])
        p2 = get_player_features(players[1])

        row = {
            "game_id": game_id,
            "source_date": SOURCE_DATE,
            "turn": turn,
            "p1_went_first": int(
                current.get("firstPlayer") == 0
            ),

            "p1_prizes_remaining": p1["prizes_remaining"],
            "p2_prizes_remaining": p2["prizes_remaining"],

            "p1_hand_size": p1["hand_size"],
            "p2_hand_size": p2["hand_size"],

            "p1_deck_size": p1["deck_size"],
            "p2_deck_size": p2["deck_size"],

            "p1_pokemon_in_play": p1["pokemon_in_play"],
            "p2_pokemon_in_play": p2["pokemon_in_play"],

            "p1_active_hp": p1["active_hp"],
            "p2_active_hp": p2["active_hp"],

            "p1_total_hp": p1["total_hp"],
            "p2_total_hp": p2["total_hp"],

            "p1_total_energy": p1["total_energy"],
            "p2_total_energy": p2["total_energy"],

            "p1_active_has_status": p1["active_has_status"],
            "p2_active_has_status": p2["active_has_status"],

            "p1_wins": p1_wins,
        }

        rows.append(row)

    return rows


with open(REPLAY_PATH, "r", encoding="utf-8") as f:
    replay = json.load(f)

rows = process_replay(replay)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if rows:
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Processed {len(rows)} turns "
        f"from game {rows[0]['game_id']}."
    )
    print(f"Saved to {OUTPUT_PATH}")

else:
    print("Replay could not be processed.")