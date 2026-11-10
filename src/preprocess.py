"""
preprocess.py — Data preprocessing for Pokémon TCG winner prediction.
Owner: Yavuz

What is in here:
  1. Load raw Pokémon TCG replay JSON files
  2. Identify the winner of each game
  3. Extract one game-state snapshot per turn
  4. Extract player features such as prizes, hand/deck size, HP, Energy, and status
  5. Skip draws or unusable replay files
  6. Combine all usable games into one processed dataset
  7. Save the processed data to data/processed/game_states.csv

Run from the repository root:
  python3 src/preprocess.py

Processed data is saved in data/processed/.
"""

import csv
import json
import re
from pathlib import Path


INPUT_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/game_states.csv")


FIELDNAMES = [
    "game_id",
    "source_date",
    "turn",
    "p1_went_first",
    "p1_prizes_remaining",
    "p2_prizes_remaining",
    "p1_hand_size",
    "p2_hand_size",
    "p1_deck_size",
    "p2_deck_size",
    "p1_pokemon_in_play",
    "p2_pokemon_in_play",
    "p1_active_hp",
    "p2_active_hp",
    "p1_total_hp",
    "p2_total_hp",
    "p1_total_energy",
    "p2_total_energy",
    "p1_active_has_status",
    "p2_active_has_status",
    "p1_wins",
]


def get_player_features(player):
    active = [
        pokemon
        for pokemon in (player.get("active") or [])
        if isinstance(pokemon, dict)
    ]

    bench = [
        pokemon
        for pokemon in (player.get("bench") or [])
        if isinstance(pokemon, dict)
    ]

    pokemon_in_play = active + bench

    if active:
        active_hp = active[0].get("hp", 0)
    else:
        active_hp = 0

    total_hp = sum(
        pokemon.get("hp", 0)
        for pokemon in pokemon_in_play
    )

    total_energy = sum(
        len(pokemon.get("energyCards") or [])
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
        "prizes_remaining": len(player.get("prize") or []),
        "hand_size": player.get("handCount", 0),
        "deck_size": player.get("deckCount", 0),
        "pokemon_in_play": len(pokemon_in_play),
        "active_hp": active_hp,
        "total_hp": total_hp,
        "total_energy": total_energy,
        "active_has_status": active_has_status,
    }


def get_winner(replay):
    rewards = replay.get("rewards")

    if rewards == [1, -1]:
        return 1

    if rewards == [-1, 1]:
        return 0

    # Draw, incomplete game, or other unusual result
    return None


def get_source_date(path):
    # Look through the folders in the path for a YYYY-MM-DD folder.
    for parent in path.parents:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", parent.name):
            return parent.name

    return "unknown"


def get_frames(replay):
    # Prefer the canonical visualization states.
    for step in replay.get("steps", []):
        for player_view in step:
            visualize = player_view.get("visualize")

            if isinstance(visualize, list) and visualize:
                return visualize

    return []


def process_replay(replay, source_date):
    p1_wins = get_winner(replay)

    if p1_wins is None:
        return []

    game_id = replay.get("info", {}).get(
        "EpisodeId",
        replay.get("id")
    )

    frames = get_frames(replay)

    if not frames:
        return []

    rows = []
    seen_turns = set()

    for frame in frames:
        current = frame.get("current")

        if not isinstance(current, dict):
            continue

        turn = current.get("turn", 0)

        # Skip setup before the actual match begins.
        if turn < 1:
            continue

        # Only keep one snapshot for each game turn.
        if turn in seen_turns:
            continue

        players = current.get("players", [])

        if len(players) != 2:
            continue

        first_player = current.get("firstPlayer")

        if first_player not in (0, 1):
            continue

        p1 = get_player_features(players[0])
        p2 = get_player_features(players[1])

        row = {
            "game_id": game_id,
            "source_date": source_date,
            "turn": turn,
            "p1_went_first": int(first_player == 0),

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
        seen_turns.add(turn)

    return rows


def main():
    json_files = sorted(INPUT_DIR.rglob("*.json"))

    print(f"Found {len(json_files)} replay files.")

    all_rows = []
    processed_games = 0
    skipped_games = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                replay = json.load(f)

            source_date = get_source_date(path)
            rows = process_replay(replay, source_date)

            if rows:
                all_rows.extend(rows)
                processed_games += 1
            else:
                print(f"Skipped unusable game: {path}")
                skipped_games += 1

        except (json.JSONDecodeError, OSError, TypeError, KeyError) as error:
            print(f"Could not process {path}: {error}")
            skipped_games += 1

    if not all_rows:
        print("No usable games were found.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Processed games: {processed_games}")
    print(f"Skipped games: {skipped_games}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()