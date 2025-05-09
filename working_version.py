from __future__ import annotations
from typing import Tuple, List, Dict, Set
from collections import deque
import copy
import random

_MOVES = [(0, -1, "W"), (-1, 0, "A"), (0, 1, "S"), (1, 0, "D")]

ENERGY_COST = {
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
    "barrel": 1
}

BASE_REWARDS = {
    "coal": 1, "deepslate coal": 1,
    "copper": 2, "deepslate copper": 2,
    "iron": 3, "deepslate iron": 3,
    "gold": 5, "deepslate gold": 5,
    "diamond": 10, "deepslate diamond": 10,
}

CHEST_EXPECTED_REWARD = 2.5 + 1.5 + 1.0 + 0.5  
BARREL_EXPECTED_ENERGY = 20 + 25*1.5 + 40*0.5 + 80*0.5  

visited_positions = set()
last_positions = deque(maxlen=10)
current_path = []
position_repetition_count = {}

ENERGY_TO_SCORE_RATIO = 0.15

def normalize_block(block) -> str:
    block = str(block).lower()
    
    for name in BASE_REWARDS:
        if name in block:
            return name
            
    if "chest" in block:
        return "chest"
    elif "barrel" in block:
        return "barrel"
    elif "deepslate" in block:
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
    return ENERGY_COST.get(name, 1)

def get_expected_reward(block) -> float:
    name = normalize_block(block)
    
    if name in BASE_REWARDS:
        return BASE_REWARDS[name]
    
    if name == "chest":
        return CHEST_EXPECTED_REWARD
    elif name == "barrel":
        return BARREL_EXPECTED_ENERGY * ENERGY_TO_SCORE_RATIO
    
    return 0

def create_map_snapshot(game_map):
    return [[normalize_block(game_map[x][y]) for y in range(len(game_map[0]))] 
            for x in range(len(game_map))]

def calculate_potential_value(game_map, position, radius=5):
    x, y = position
    width, height = len(game_map), len(game_map[0])
    total_value = 0
    
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                block = normalize_block(game_map[nx][ny])
                dist = abs(dx) + abs(dy)
                if dist > 0:  
                    block_value = get_expected_reward(block)
                    total_value += block_value / (dist * 1.5)
    
    return total_value

def agent_logic(game_map: List[List[str]], position: Tuple[int, int], energy: int) -> str:
    global last_positions, position_repetition_count, current_path

    position_str = f"{position[0]},{position[1]}"
    position_repetition_count[position_str] = position_repetition_count.get(position_str, 0) + 1
    last_positions.append(position)

    if current_path:
        next_move = current_path.pop(0)
        dx, dy, action = _MOVES[next_move]
        nx, ny = position[0] + dx, position[1] + dy

        if (0 <= nx < len(game_map) and 0 <= ny < len(game_map[0]) and 
            energy >= get_cost(game_map[nx][ny])):
            return action
        else:
            current_path = []

    best_action = 'I'  
    best_value = float('-inf')
    
    for move_idx, (dx, dy, action) in enumerate(_MOVES):
        nx, ny = position[0] + dx, position[1] + dy
        
        if not (0 <= nx < len(game_map) and 0 <= ny < len(game_map[0])):
            continue
            
        next_pos = (nx, ny)
        cost = get_cost(game_map[nx][ny])
  
        if energy < cost:
            continue

        reward = get_expected_reward(game_map[nx][ny])

        repetition_penalty = 0
        next_pos_str = f"{nx},{ny}"
        if next_pos_str in position_repetition_count:
            repetition_penalty = min(position_repetition_count[next_pos_str] * 0.5, 3.0)

        potential = calculate_potential_value(game_map, next_pos)

        remaining_energy = energy - cost

        future_value = expectimax(game_map, next_pos, remaining_energy, depth=3)

        total_value = (reward + future_value + potential*0.3) - repetition_penalty
        
        if total_value > best_value:
            best_value = total_value
            best_action = action
    
    if best_action == 'I' and energy > 10:
        path = find_path_to_valuable_target(game_map, position, energy)
        if path:
            current_path = path[1:] 
            return path[0]
    
    return best_action

def expectimax(game_map: List[List[str]], position: Tuple[int, int], energy: int, depth: int) -> float:
    if depth == 0 or energy <= 1: 
        return 0
    
    max_value = 0
    total_moves = 0
    total_value = 0
    

    for dx, dy, _ in _MOVES:
        nx, ny = position[0] + dx, position[1] + dy
        
        if not (0 <= nx < len(game_map) and 0 <= ny < len(game_map[0])):
            continue
            
        cost = get_cost(game_map[nx][ny])

        if energy < cost:
            continue
            
        total_moves += 1
        reward = get_expected_reward(game_map[nx][ny])
        remaining_energy = energy - cost

        if normalize_block(game_map[nx][ny]) in ["chest", "barrel"]:
            future_value = expectimax(game_map, (nx, ny), remaining_energy, depth - 1)
            value = reward + future_value * 0.8 
        else:
            future_value = expectimax(game_map, (nx, ny), remaining_energy, depth - 1)
            value = reward + future_value

        max_value = max(max_value, value)
        
        total_value += value

    if total_moves == 0:
        return 0

    return max_value * 0.7 + (total_value / total_moves) * 0.3

def find_path_to_valuable_target(game_map, position, energy, max_depth=15):
    """Find a path to a valuable target using BFS."""
    queue = deque([(position, [], energy)])
    visited = {position}
    best_path = None
    best_value = -1
    
    while queue and len(visited) < 100: 
        pos, path, remaining_energy = queue.popleft()
        x, y = pos

        block = normalize_block(game_map[x][y])
        value = get_expected_reward(game_map[x][y])
        
        area_value = calculate_potential_value(game_map, pos, radius=3)

        position_value = value + area_value - (len(path) * 0.5)
    
        if position_value > best_value and path:
            best_value = position_value
            best_path = path
        
        if len(path) >= max_depth:
            continue

        for i, (dx, dy, action) in enumerate(_MOVES):
            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
 
            if not (0 <= nx < len(game_map) and 0 <= ny < len(game_map[0])):
                continue

            if next_pos in visited:
                continue
                
            cost = get_cost(game_map[nx][ny])
            
            if remaining_energy < cost:
                continue

            new_path = path + [action]
            queue.append((next_pos, new_path, remaining_energy - cost))
            visited.add(next_pos)
    
    return best_path

def action_to_index(action):
    for i, (_, _, a) in enumerate(_MOVES):
        if a == action:
            return i
    return -1