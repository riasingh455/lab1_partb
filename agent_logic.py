from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional
from collections import deque
import heapq
import random

DIRECTIONS = [(0, -1, "W"), (-1, 0, "A"), (0, 1, "S"), (1, 0, "D")]

LOOT_REWARD = {
    "coal": 1, "deepslate coal": 1,
    "copper": 2, "deepslate copper": 2,
    "iron": 3, "deepslate iron": 3,
    "gold": 5, "deepslate gold": 5,
    "diamond": 10, "deepslate diamond": 10,
    "chest": 2.5 * 2 + 1.5 * 3 + 1 * 5 + 0.5 * 10,
    "barrel": 1 * 20 + 1.5 * 25 + 0.5 * 40 + 0.5 * 80,
}

ENERGY_TABLE = {
    "empty": 1,
    "dirt": 2,
    "stone": 4,
    "deepslate": 10,
    "coal": 4, "deepslate coal": 10,
    "copper": 4, "deepslate copper": 10,
    "iron": 4, "deepslate iron": 10,
    "gold": 4, "deepslate gold": 10,
    "diamond": 4, "deepslate diamond": 10,
    "chest": 1,
    "barrel": 1,
}

ENERGY_SCORE_FACTOR = 0.2

visited_cells: Set[Tuple[int, int]] = set()
recent_positions: deque = deque()
action_history: deque = deque()
planned_path: List[str] = []
current_goal: Optional[Tuple[int, int]] = None
ore_cells: List[Tuple[int, int, int]] = []
chest_cells: List[Tuple[int, int]] = []
barrel_cells: List[Tuple[int, int]] = []
is_first_call = True
world_map = None


def normalize_block(block) -> str:
    block = str(block).lower()

    if "chest" in block:
        return "chest"
    if "barrel" in block:
        return "barrel"

    for ore_type in ["diamond", "gold", "iron", "copper", "coal"]:
        if ore_type in block:
            if "deepslate" in block:
                return f"deepslate {ore_type}"
            return ore_type
    if "deepslate" in block:
        return "deepslate"
    elif "stone" in block:
        return "stone"
    elif "dirt" in block:
        return "dirt"
    elif "empty" in block:
        return "empty"
    elif "boundary" in block:
        return "boundary"
    return "empty"


def get_cost(block) -> int:
    name = normalize_block(block)
    return ENERGY_TABLE.get(name, 1)


def get_expected_reward(block) -> float:
    name = normalize_block(block)
    reward = LOOT_REWARD.get(name, 0)
    if name == "barrel":
        return reward * ENERGY_SCORE_FACTOR
    return reward


def update_internal_state(game_map, position):
    global world_map, ore_cells, chest_cells, barrel_cells, is_first_call, visited_cells

    if is_first_call:
        width, height = len(game_map), len(game_map[0])
        world_map = [[None for _ in range(height)] for _ in range(width)]
        ore_cells = []
        chest_cells = []
        barrel_cells = []
        is_first_call = False

        for x in range(width):
            for y in range(height):
                #check pass
                block_name = normalize_block(game_map[x][y])

                if block_name in LOOT_REWARD and LOOT_REWARD[block_name] > 0:
                    if block_name == "chest":
                        chest_cells.append((x, y))
                    elif block_name == "barrel":
                        barrel_cells.append((x, y))
                    elif block_name not in ["empty", "dirt", "stone", "deepslate"]:
                        ore_cells.append((x, y, LOOT_REWARD[block_name]))

    x, y = position
    visited_cells.add(position)

    if position == current_goal:
        #check pass 2
        block_name = normalize_block(game_map[x][y])
        if position in chest_cells:
            chest_cells.remove(position)
        elif position in barrel_cells:
            barrel_cells.remove(position)
        else:
            ore_cells = [pos for pos in ore_cells if (pos[0], pos[1]) != position]


def estimate_path_cost(game_map, start, end):
    width, height = len(game_map), len(game_map[0])

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    open_set = [(0, start)]
    heapq.heapify(open_set)

    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            return g_score[current]

        x, y = current
        for dx, dy, _ in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue

            next_pos = (nx, ny)
            #check 4
            energy_cost = get_cost(game_map[nx][ny])

            new_g = g_score[current] + energy_cost

            if next_pos not in g_score or new_g < g_score[next_pos]:
                g_score[next_pos] = new_g
                h = abs(nx - end[0]) + abs(ny - end[1])
                f = new_g + h
                heapq.heappush(open_set, (f, next_pos))
                came_from[next_pos] = current
    return float("inf")


def find_path_to_target(game_map, start, target, energy):
    width, height = len(game_map), len(game_map[0])

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    open_set = [(0, start)]
    heapq.heapify(open_set)

    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            while current in came_from:
                current, action = came_from[current]
                path.append(action)
            return path[::-1]

        x, y = current
        for dx, dy, action in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue

            next_pos = (nx, ny)
            #check pass 3
            energy_cost = get_cost(game_map[nx][ny])

            new_g = g_score[current] + energy_cost

            if new_g > energy:
                continue

            if next_pos not in g_score or new_g < g_score[next_pos]:
                g_score[next_pos] = new_g
                h = abs(nx - target[0]) + abs(ny - target[1])
                f = new_g + h
                heapq.heappush(open_set, (f, next_pos))
                came_from[next_pos] = (current, action)

    return []


def select_target(game_map, position, energy):

    potential_targets = []
    x, y = position

    for ore_x, ore_y, reward in ore_cells:
        target = (ore_x, ore_y)
        path_cost = estimate_path_cost(game_map, position, target)

        if path_cost <= energy and path_cost < float("inf"):
            utility = reward / max(1, path_cost)
            potential_targets.append((target, utility, reward, "ore"))
    for chest_x, chest_y in chest_cells:
        target = (chest_x, chest_y)
        path_cost = estimate_path_cost(game_map, position, target)

        if path_cost <= energy and path_cost < float("inf"):
            reward_val = LOOT_REWARD["chest"]
            utility = reward_val / max(1, path_cost)
            potential_targets.append((target, utility, reward_val, "chest"))

    for barrel_x, barrel_y in barrel_cells:
        target = (barrel_x, barrel_y)
        path_cost = estimate_path_cost(game_map, position, target)

        if path_cost <= energy and path_cost < float("inf"):
            energy_value = LOOT_REWARD["barrel"]
            energy_factor = max(1, 100 / max(10, energy))
            adjusted_reward = energy_value * ENERGY_SCORE_FACTOR * energy_factor
            utility = adjusted_reward / max(1, path_cost)

            potential_targets.append((target, utility, adjusted_reward, "barrel"))

    potential_targets.sort(key=lambda item: item[1], reverse=True)

    if potential_targets:
        return potential_targets[0][0]
    return None


def expectimax(game_map, position, energy, depth=3):
    if depth == 0 or energy <= 0:
        return 0

    x, y = position
    width, height = len(game_map), len(game_map[0])
    max_value = 0

    for dx, dy, _ in DIRECTIONS:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < width and 0 <= ny < height):
            continue

        next_pos = (nx, ny)

        if next_pos in recent_positions and depth > 1:
            continue
        #check 5
        cost = get_cost(game_map[nx][ny])
        if energy < cost:
            continue
        #check 6
        reward = get_expected_reward(game_map[nx][ny])
        remaining_energy = energy - cost

        #check 7

        if "chest" in str(game_map[nx][ny]).lower() or "barrel" in str(
            game_map[nx][ny]
        ).lower():
            #check 8
            if "barrel" in str(game_map[nx][ny]).lower():
                energy_factor = max(1, 100 / max(10, energy))
                reward *= energy_factor

            future_value = expectimax(game_map, next_pos, remaining_energy, depth - 1)
            value = reward + future_value
        else:
            future_value = expectimax(game_map, next_pos, remaining_energy, depth - 1)
            value = reward + future_value

        max_value = max(max_value, value)

    return max_value


def detect_oscillation():
    if len(action_history) >= 6:
        pattern1 = list(action_history)[-6:] == list(action_history)[-6:-4] * 3
        pattern2 = list(action_history)[-4:] == list(action_history)[-4:-2] * 2
        return pattern1 or pattern2
    return False


def agent_logic(game_map: List[List[str]], position: Tuple[int, int], energy: int) -> str:
    """
    Main agent logic function that decides the next move

    Args:
        game_map: 2D list representing the game map
        position: Current position (x, y) of the player
        energy: Current energy of the player

    Returns:
        Direction to move: 'W' (up), 'A' (left), 'S' (down), 'D' (right), or 'I' (stay)
    """
    try:
        global recent_positions, planned_path, current_goal, action_history

        update_internal_state(game_map, position)
        recent_positions.append(position)

        if len(planned_path) > 0:
            try:
                action = planned_path.pop(0)
            except Exception as e:
                print("It's that damn pop!")
                print(e)
            action_history.append(action)
            return action
        try:
            if not current_goal or position == current_goal:
                current_goal = select_target(game_map, position, energy)
                if current_goal:
                    planned_path = find_path_to_target(
                        game_map, position, current_goal, energy
                    )
                    if planned_path:
                        action = planned_path.pop(0)
                        action_history.append(action)
                        return action
        except Exception as e:
            print("yo its this block")
            print(e)
        try:
            # if detect_oscillation():
            #     actions = ["W", "A", "S", "D", "I"]
            #     action = random.choice(actions)
            #     action_history.append(action)
            #     return action
            def detect_oscillation() -> bool:
                hist = list(action_history)  # ← convert once
                if len(hist) >= 6:
                    if hist[-6:] == hist[-6:-4] * 3:
                        return True
                if len(hist) >= 4:
                    if hist[-4:] == hist[-4:-2] * 2:
                        return True
                return False
        except Exception as e:
            print("no its this one")
            print(e)

        best_action = "I"
        best_value = float("-inf")

        for dx, dy, action in DIRECTIONS:
            nx, ny = position[0] + dx, position[1] + dy
            if not (0 <= nx < len(game_map) and 0 <= ny < len(game_map[0])):
                continue

            next_pos = (nx, ny)
            revisit_penalty = 5 if next_pos in recent_positions else 0

            #check 9
            cost = get_cost(game_map[nx][ny])
            if energy < cost:
                continue

            reward = get_expected_reward(game_map[nx][ny])
            remaining_energy = energy - cost

            if "barrel" in str(game_map[nx][ny]).lower():
                energy_factor = max(1, 100 / max(10, energy))
                reward *= energy_factor

            future_value = expectimax(game_map, next_pos, remaining_energy, depth=3)
            total_value = reward + future_value - revisit_penalty

            if total_value > best_value:
                best_value = total_value
                best_action = action
    except Exception as e:
        print("Did this die lmao")
        print(e)

    action_history.append(best_action)

    return best_action
