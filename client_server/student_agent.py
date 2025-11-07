import random
import copy
from typing import List, Dict, Any, Optional, Tuple, Set, NamedTuple
from collections import deque
import heapq
import time
turn_count = 0
remaining_time = 0
initial_time = -1


def switch_player(current: str) -> str:
    
    if current == "circle":
        return "square"
    return "circle"
def get_scoring_columns(width: int) -> List[int]:
    
    scoring_width = 4
    beginning = max(0, (width - scoring_width) // 2)
    column_list = []
    idx = 0
    while idx < scoring_width:
        column_list.append(beginning + idx)
        idx += 1
    return column_list

def circle_goal_line() -> int:
   
    return 2


def within_grid(col: int, row: int, height: int, width: int) -> bool:
    
    if col < 0 or col >= width:
        return False
    if row < 0 or row >= height:
        return False
    return True
def square_goal_line(height: int) -> int:
    
    return height - 3
def check_enemy_goal(col: int, row: int, current_player: str, height: int, width: int, goal_cols: List[int]) -> bool:
    
    if current_player != "circle":
        goal_row = circle_goal_line()
    else:
        goal_row = square_goal_line(height)
    return (row == goal_row) and (col in goal_cols)

def check_player_goal(col: int, row: int, current_player: str, height: int, width: int, goal_cols: List[int]) -> bool:
    
    if current_player != "square":
        goal_row = circle_goal_line()
    else:
        goal_row = square_goal_line(height)
    return (row == goal_row) and (col in goal_cols)



class GameToken:
    def __init__(self, token_owner: str, token_type: str = "stone", direction: Optional[str] = None):
        self.owner = token_owner
        self.side = token_type
        self.orientation = direction if direction else "horizontal"

    
    
    

def calculate_river_paths(grid: List[List[Optional[GameToken]]],
                         river_col: int, river_row: int, 
                         start_col: int, start_row: int, 
                         current_player: str,
                         height: int, width: int, goal_cols: List[int],
                         pushing_river: bool = False) -> List[Tuple[int, int]]:
    
    reachable = []
    explored = set()
    to_explore = deque([(river_col, river_row)])
    
    while len(to_explore) > 0:
        col, row = to_explore.popleft()
        
        if (col, row) in explored:
            continue
        if not within_grid(col, row, height, width):
            continue
            
        explored.add((col, row))
        
        token = grid[row][col]
        if pushing_river and col == river_col and row == river_row:
            token = grid[start_row][start_col]
        
        if not token:
            if not check_enemy_goal(col, row, current_player, height, width, goal_cols):
                reachable.append((col, row))
            continue
        
        if token.side != "river":
            continue
        
        if token.orientation != "vertical":
            flow_directions = [(1, 0), (-1, 0)]
        else:
            flow_directions = [(0, 1), (0, -1)]
        
        for dc, dr in flow_directions:
            next_col = col + dc
            next_row = row + dr
            
            while within_grid(next_col, next_row, height, width):
                if check_enemy_goal(next_col, next_row, current_player, height, width, goal_cols):
                    break
                    
                next_token = grid[next_row][next_col]
                
                if not next_token:
                    reachable.append((next_col, next_row))
                    next_col += dc
                    next_row += dr
                elif next_col == start_col and next_row == start_row:
                    next_col += dc
                    next_row += dr
                elif next_token.side == "river":
                    to_explore.append((next_col, next_row))
                    break
                else:
                    break
    
    
    unique_positions = []
    found = set()
    for pos in reachable:
        if pos not in found:
            found.add(pos)
            unique_positions.append(pos)
    
    return unique_positions

def _create_move_action(from_col: int, from_row: int, to_col: int, to_row: int) -> Dict[str, Any]:
    
    return {
        "action": "move",
        "from": [from_col, from_row],
        "to": [to_col, to_row]
    }

def _create_push_action(from_col: int, from_row: int, to_col: int, to_row: int, 
                       push_col: int, push_row: int, is_river_push: bool = False) -> Dict[str, Any]:
    
    action = {
        "action": "push",
        "from": [from_col, from_row],
        "to": [to_col, to_row],
        "pushed_to": [push_col, push_row]
    }
    if is_river_push:
        action["river_push"] = True
    return action

def _create_flip_action(col: int, row: int, orientation: Optional[str] = None) -> Dict[str, Any]:
    
    action = {
        "action": "flip",
        "from": [col, row]
    }
    if orientation:
        action["orientation"] = orientation
    return action

def _create_rotate_action(col: int, row: int) -> Dict[str, Any]:
    
    return {
        "action": "rotate",
        "from": [col, row]
    }

def _generate_stone_moves(grid: List[List[Optional[GameToken]]], col: int, row: int,
                         current_player: str, height: int, width: int, 
                         goal_cols: List[int], movements: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    
    actions = []
    
    for dc, dr in movements:
        new_col = col + dc
        new_row = row + dr
        
        if not within_grid(new_col, new_row, height, width):
            continue
        if check_enemy_goal(new_col, new_row, current_player, height, width, goal_cols):
            continue
            
        target = grid[new_row][new_col]
        
        if not target:
            
            actions.append(_create_move_action(col, row, new_col, new_row))
        elif target.side == "river":
            
            paths = calculate_river_paths(grid, new_col, new_row, col, row, 
                                        current_player, height, width, goal_cols)
            for destination in paths:
                actions.append(_create_move_action(col, row, destination[0], destination[1]))
        else:
            
            push_col = new_col + dc
            push_row = new_row + dr
            
            if (within_grid(push_col, push_row, height, width) and
                not grid[push_row][push_col] and
                not check_enemy_goal(push_col, push_row, target.owner, height, width, goal_cols)):
                
                actions.append(_create_push_action(col, row, new_col, new_row, push_col, push_row))
    
    return actions

def _generate_stone_transformations(col: int, row: int) -> List[Dict[str, Any]]:
    
    return [
        _create_flip_action(col, row, "horizontal"),
        _create_flip_action(col, row, "vertical")
    ]

def _generate_river_basic_moves(grid: List[List[Optional[GameToken]]], col: int, row: int,
                               current_player: str, height: int, width: int,
                               goal_cols: List[int], movements: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    
    actions = []
    
    for dc, dr in movements:
        new_col = col + dc
        new_row = row + dr
        
        if not within_grid(new_col, new_row, height, width):
            continue
        if check_enemy_goal(new_col, new_row, current_player, height, width, goal_cols):
            continue
            
        target = grid[new_row][new_col]
        
        if not target:
            
            actions.append(_create_move_action(col, row, new_col, new_row))
        elif target.side == "river":
            
            paths = calculate_river_paths(grid, new_col, new_row, col, row,
                                        current_player, height, width, goal_cols)
            for destination in paths:
                actions.append(_create_move_action(col, row, destination[0], destination[1]))
    
    return actions

def _get_river_flow_directions(orientation: str) -> List[Tuple[int, int]]:
    
    if orientation == "horizontal":
        return [(1, 0), (-1, 0)]
    else:  
        return [(0, 1), (0, -1)]

def _generate_river_push_actions(grid: List[List[Optional[GameToken]]], col: int, row: int,
                                token: GameToken, dc: int, dr: int, new_col: int, new_row: int,
                                target: GameToken, height: int, width: int, goal_cols: List[int]) -> List[Dict[str, Any]]:
    
    actions = []
    flow_directions = _get_river_flow_directions(token.orientation)
    
    
    for fd_col, fd_row in flow_directions:
        if dc == fd_col and dr == fd_row:
            
            push_col = new_col + fd_col
            push_row = new_row + fd_row
            
            
            while (within_grid(push_col, push_row, height, width) and
                   not grid[push_row][push_col] and
                   not check_enemy_goal(push_col, push_row, target.owner, height, width, goal_cols)):
                
                actions.append(_create_push_action(col, row, new_col, new_row, 
                                                 push_col, push_row, is_river_push=True))
                
                
                push_col += fd_col
                push_row += fd_row
            
            break  
    
    return actions

def _generate_river_actions(grid: List[List[Optional[GameToken]]], col: int, row: int,
                           token: GameToken, current_player: str, height: int, width: int,
                           goal_cols: List[int], movements: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    
    actions = []
    
    
    actions.extend(_generate_river_basic_moves(grid, col, row, current_player, 
                                              height, width, goal_cols, movements))
    
    
    for dc, dr in movements:
        new_col = col + dc
        new_row = row + dr
        
        if not within_grid(new_col, new_row, height, width):
            continue
        if check_enemy_goal(new_col, new_row, current_player, height, width, goal_cols):
            continue
            
        target = grid[new_row][new_col]
        
        if target and target.side == "stone":
            
            actions.extend(_generate_river_push_actions(grid, col, row, token, dc, dr,
                                                       new_col, new_row, target,
                                                       height, width, goal_cols))
    
   
    actions.append(_create_flip_action(col, row))
    actions.append(_create_rotate_action(col, row))
    
    return actions

def _get_player_pieces(grid: List[List[Optional[GameToken]]], current_player: str,
                      height: int, width: int) -> List[Tuple[int, int, GameToken]]:
    
    player_pieces = []
    
    for row in range(height):
        for col in range(width):
            token = grid[row][col]
            if token and token.owner == current_player:
                player_pieces.append((col, row, token))
    
    return player_pieces

def create_all_actions(grid: List[List[Optional[GameToken]]],
                      current_player: str, height: int, width: int, 
                      goal_cols: List[int]) -> List[Dict[str, Any]]:
    
    action_list = []
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
   
    player_pieces = _get_player_pieces(grid, current_player, height, width)
    
    
    for col, row, token in player_pieces:
        if token.side != "river":  
            action_list.extend(_generate_stone_moves(grid, col, row, current_player,
                                                   height, width, goal_cols, movements))
            
            action_list.extend(_generate_stone_transformations(col, row))
        else:  
            
            action_list.extend(_generate_river_actions(grid, col, row, token,
                                                      current_player, height, width,
                                                      goal_cols, movements))
    
    return action_list



def _create_grid_copy(grid: List[List[Any]]) -> List[List[Any]]:
    
    return copy.deepcopy(grid)

def _extract_position_coords(position: List[int]) -> Tuple[int, int]:
    
    return position[0], position[1]

def _validate_piece_ownership(piece: Any, player_name: str) -> Tuple[bool, str]:
    
    if not piece:
        return False, "Invalid piece"
    if piece.owner != player_name:
        return False, "Invalid piece"
    return True, ""

def _validate_move_destination(to_col: int, to_row: int, grid_copy: List[List[Any]], 
                             player_name: str, height: int, width: int, 
                             goal_cols: List[int]) -> Tuple[bool, str]:
    
    if not within_grid(to_col, to_row, height, width):
        return False, "Out of bounds"
    
    if check_enemy_goal(to_col, to_row, player_name, height, width, goal_cols):
        return False, "Cannot move to opponent's score area"
    
    if grid_copy[to_row][to_col] is not None:
        return False, "Destination occupied"
    
    return True, ""

def _execute_move_action(grid_copy: List[List[Any]], piece: Any, 
                        from_col: int, from_row: int, 
                        to_col: int, to_row: int) -> None:
    
    grid_copy[to_row][to_col] = piece
    grid_copy[from_row][from_col] = None

def _handle_move_action(action: Dict[str, Any], grid_copy: List[List[Any]], 
                       piece: Any, from_col: int, from_row: int,
                       player_name: str, height: int, width: int, 
                       goal_cols: List[int]) -> Tuple[bool, Any]:
    
    to_col, to_row = _extract_position_coords(action["to"])
    
    valid, error_msg = _validate_move_destination(to_col, to_row, grid_copy, 
                                                 player_name, height, width, goal_cols)
    if not valid:
        return False, error_msg
    
    _execute_move_action(grid_copy, piece, from_col, from_row, to_col, to_row)
    return True, grid_copy

def _validate_push_positions(to_col: int, to_row: int, push_col: int, push_row: int,
                           height: int, width: int) -> Tuple[bool, str]:
    
    if not within_grid(to_col, to_row, height, width):
        return False, "Push target out of bounds"
    if not within_grid(push_col, push_row, height, width):
        return False, "Push destination out of bounds"
    return True, ""

def _validate_push_target(target_piece: Any, piece: Any, grid_copy: List[List[Any]], 
                         push_row: int, push_col: int, height: int, width: int,
                         goal_cols: List[int]) -> Tuple[bool, str]:
    
    if not target_piece:
        return False, "No piece to push"
    
    if piece.side == "river" and target_piece.side != "stone":
        return False, "River can only push stone"
    
    if grid_copy[push_row][push_col] is not None:
        return False, "Push destination occupied"
    
    if check_enemy_goal(push_col, push_row, target_piece.owner, height, width, goal_cols):
        return False, "Cannot push to opponent's score area"
    
    return True, ""

def _execute_push_action(grid_copy: List[List[Any]], piece: Any, target_piece: Any,
                        from_col: int, from_row: int, to_col: int, to_row: int,
                        push_col: int, push_row: int, is_river_push: bool) -> None:
    
    
    grid_copy[push_row][push_col] = target_piece
    
    grid_copy[to_row][to_col] = piece
    
    grid_copy[from_row][from_col] = None
    
    
    if is_river_push and piece.side == "river":
        grid_copy[to_row][to_col] = GameToken(piece.owner, "stone", None)

def _handle_push_action(action: Dict[str, Any], grid_copy: List[List[Any]], 
                       piece: Any, from_col: int, from_row: int,
                       height: int, width: int, goal_cols: List[int]) -> Tuple[bool, Any]:
    
    to_col, to_row = _extract_position_coords(action["to"])
    push_col, push_row = _extract_position_coords(action["pushed_to"])
    
   
    valid, error_msg = _validate_push_positions(to_col, to_row, push_col, push_row, 
                                               height, width)
    if not valid:
        return False, error_msg
    
    target_piece = grid_copy[to_row][to_col]
    
    
    valid, error_msg = _validate_push_target(target_piece, piece, grid_copy, 
                                           push_row, push_col, height, width, goal_cols)
    if not valid:
        return False, error_msg
    
    
    is_river_push = action.get("river_push", False)
    _execute_push_action(grid_copy, piece, target_piece, from_col, from_row,
                        to_col, to_row, push_col, push_row, is_river_push)
    
    return True, grid_copy

def _create_flipped_piece(piece: Any, orientation: Optional[str] = None) -> Any:
    
    if piece.side == "stone":
        
        return GameToken(piece.owner, "river", orientation or "horizontal")
    else:
        
        return GameToken(piece.owner, "stone", None)

def _handle_flip_action(action: Dict[str, Any], grid_copy: List[List[Any]], 
                       piece: Any, from_col: int, from_row: int) -> Tuple[bool, Any]:
    
    orientation = action.get("orientation")
    new_piece = _create_flipped_piece(piece, orientation)
    grid_copy[from_row][from_col] = new_piece
    return True, grid_copy

def _create_rotated_river(piece: Any) -> Any:
    
    new_orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
    return GameToken(piece.owner, "river", new_orientation)

def _handle_rotate_action(grid_copy: List[List[Any]], piece: Any, 
                         from_col: int, from_row: int) -> Tuple[bool, Any]:
    
    if piece.side != "river":
        return False, "Can only rotate rivers"
    
    new_piece = _create_rotated_river(piece)
    grid_copy[from_row][from_col] = new_piece
    return True, grid_copy

def _apply_action_with_engine(action: Dict[str, Any], grid: List[List[Any]], 
                            player_name: str, height: int, width: int, 
                            goal_cols: List[int]) -> Tuple[bool, Any]:
    
    from gameEngine import validate_and_apply_move
    
    grid_copy = _create_grid_copy(grid)
    
    
    if action.get("river_push", False):
        from_col, from_row = _extract_position_coords(action["from"])
        original_piece = grid_copy[from_row][from_col]
    
    success, message = validate_and_apply_move(grid_copy, action, player_name, 
                                             height, width, goal_cols)
    
    
    if success and action.get("river_push", False) and original_piece.side == "river":
        grid_copy[from_row][from_col] = GameToken(original_piece.owner, "stone", None)
    
    return success, grid_copy if success else message

def _apply_action_fallback(action: Dict[str, Any], grid: List[List[Any]], 
                         player_name: str, height: int, width: int, 
                         goal_cols: List[int]) -> Tuple[bool, Any]:
    
    grid_copy = _create_grid_copy(grid)
    
    try:
        action_type = action["action"]
        from_col, from_row = _extract_position_coords(action["from"])
        
        piece = grid_copy[from_row][from_col]
        
        
        valid, error_msg = _validate_piece_ownership(piece, player_name)
        if not valid:
            return False, error_msg
        
       
        if action_type == "move":
            return _handle_move_action(action, grid_copy, piece, from_col, from_row,
                                     player_name, height, width, goal_cols)
        
        elif action_type == "push":
            return _handle_push_action(action, grid_copy, piece, from_col, from_row,
                                     height, width, goal_cols)
        
        elif action_type == "flip":
            return _handle_flip_action(action, grid_copy, piece, from_col, from_row)
        
        elif action_type == "rotate":
            return _handle_rotate_action(grid_copy, piece, from_col, from_row)
        
        else:
            return False, "Unknown action type"
            
    except Exception as e:
        return False, f"Error applying action: {str(e)}"

def apply_action(grid: List[List[Any]], action: Dict[str, Any], 
                player_name: str, height: int, width: int, 
                goal_cols: List[int]) -> Tuple[bool, Any]:
    
    try:
        return _apply_action_with_engine(action, grid, player_name, height, width, goal_cols)
    except ImportError:
        return _apply_action_fallback(action, grid, player_name, height, width, goal_cols)

def compute_grid_distance(grid, x_pos: int, y_pos: int, player_name: str, 
                         height: int, width: int) -> int:
    
    target_row = circle_goal_line() if player_name == "circle" else square_goal_line(height)
    
    col_start = max(0, (width - 4) // 2)
    col_end = col_start + 4
    
    leftmost = col_start
    rightmost = col_end
    
    idx = col_start
    while idx <= col_end:
        if idx < width and grid[target_row][idx] is None:
            leftmost = idx
            break
        idx += 1
    
    idx = col_end - 1
    while idx >= col_start:
        if idx >= 0 and grid[target_row][idx] is None:
            rightmost = idx
            break
        idx -= 1
    
    if x_pos < leftmost:
        x_target = leftmost
    elif x_pos > rightmost - 1:
        x_target = rightmost - 1
    else:
        x_target = x_pos
    
    return abs(x_pos - x_target) + abs(y_pos - target_row)

def aggregate_distances(grid: List[List[Any]], player_name: str, 
                       height: int, width: int, goal_cols: List[int]) -> Tuple[int, int]:
    
    player_sum = 0
    enemy_sum = 0
    enemy = switch_player(player_name)
    
    y = 0
    while y < height:
        x = 0
        while x < width:
            token = grid[y][x]
            if token:
                if token.owner == player_name:
                    player_sum += pow(compute_grid_distance(grid, x, y, player_name, height, width), 0.5)
                else:
                    enemy_sum += pow(compute_grid_distance(grid, x, y, enemy, height, width), 0.5)
            x += 1
        y += 1
    
    return player_sum, enemy_sum

def closest_reachable_distance(grid: List[List[Any]], player_name: str, 
                              height: int, width: int, goal_cols: List[int]) -> List[int]:
    
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    distance_map = {}
    
    row_idx = 0
    while row_idx < height:
        col_idx = 0
        while col_idx < width:
            min_dist = compute_grid_distance(grid, col_idx, row_idx, player_name, height, width)
            token = grid[row_idx][col_idx]
            
            if token and token.owner == player_name:
                if token.side == "stone":
                    move_idx = 0
                    while move_idx < len(movements):
                        dc, dr = movements[move_idx]
                        new_col = col_idx + dc
                        new_row = row_idx + dr
                        
                        if within_grid(new_col, new_row, height, width):
                            if check_enemy_goal(new_col, new_row, player_name, height, width, goal_cols):
                                min_dist = 0
                                break
                            
                            target = grid[new_row][new_col]
                            if not target:
                                min_dist = min(min_dist, compute_grid_distance(grid, new_col, new_row, player_name, height, width))
                            elif target.side == "river":
                                paths = calculate_river_paths(grid, new_col, new_row, col_idx, row_idx, 
                                                            player_name, height, width, goal_cols)
                                for dest in paths:
                                    min_dist = min(min_dist, compute_grid_distance(grid, dest[0], dest[1], player_name, height, width))
                            else:
                                push_col = new_col + dc
                                push_row = new_row + dr
                                if (within_grid(push_col, push_row, height, width) and 
                                    not grid[push_row][push_col] and
                                    not check_enemy_goal(push_col, push_row, target.owner, height, width, goal_cols)):
                                    min_dist = min(min_dist, compute_grid_distance(grid, new_col, new_row, player_name, height, width))
                                    if target.owner == player_name:
                                        key = (new_col, new_row)
                                        current = distance_map.get(key, compute_grid_distance(grid, new_col, new_row, player_name, height, width))
                                        distance_map[key] = min(current, compute_grid_distance(grid, push_col, push_row, player_name, height, width))
                        
                        move_idx += 1
                else:  
                    move_idx = 0
                    while move_idx < len(movements):
                        dc, dr = movements[move_idx]
                        new_col = col_idx + dc
                        new_row = row_idx + dr
                        
                        if within_grid(new_col, new_row, height, width):
                            if check_enemy_goal(new_col, new_row, player_name, height, width, goal_cols):
                                min_dist = 0
                                break
                            
                            target = grid[new_row][new_col]
                            if not target:
                                min_dist = min(min_dist, compute_grid_distance(grid, new_col, new_row, player_name, height, width))
                            elif target.side == "river":
                                paths = calculate_river_paths(grid, new_col, new_row, col_idx, row_idx, 
                                                            player_name, height, width, goal_cols)
                                for dest in paths:
                                    min_dist = min(min_dist, compute_grid_distance(grid, dest[0], dest[1], player_name, height, width))
                            else:
                                push_col = new_col + dc
                                push_row = new_row + dr
                                if (within_grid(push_col, push_row, height, width) and 
                                    not grid[push_row][push_col] and
                                    not check_enemy_goal(push_col, push_row, target.owner, height, width, goal_cols)):
                                    min_dist = min(min_dist, compute_grid_distance(grid, push_col, push_row, player_name, height, width))
                        
                        move_idx += 1
                
                distance_map[(col_idx, row_idx)] = min_dist
            
            col_idx += 1
        row_idx += 1
    
    sorted_distances = sorted(distance_map.items(), key=lambda x: x[1])
    return [dist[1] for dist in sorted_distances]

def pathfinding_to_goal(grid, player_name, height, width, goal_cols, depth_limit):
    
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    total_cost = 0
    cost_list = []
    
    for row in range(height):
        for col in range(width):
            token = grid[row][col]
            if not token or token.owner != player_name:
                continue
            
            queue_idx = 0
            search_queue = [(col, row)]
            total_cost += height + width
            
            distance = {(col, row): 0}
            
            while queue_idx < len(search_queue):
                curr_col, curr_row = search_queue[queue_idx]
                queue_idx += 1
                
                if distance[(curr_col, curr_row)] > depth_limit:
                    break
                
                if check_player_goal(curr_col, curr_row, player_name, height, width, goal_cols):
                    total_cost += distance[(curr_col, curr_row)]
                    if distance[(curr_col, curr_row)] == 1:
                        total_cost -= 200
                    total_cost -= height - width
                    cost_list.append(distance[(curr_col, curr_row)])
                    break
                
                
                move_idx = 0
                while move_idx < len(movements):
                    dc, dr = movements[move_idx]
                    next_col = curr_col + dc
                    next_row = curr_row + dr
                    
                    if within_grid(next_col, next_row, height, width) and (next_col, next_row) not in distance:
                        distance[(next_col, next_row)] = distance[(curr_col, curr_row)] + 1
                        if not grid[next_row][next_col] or grid[next_row][next_col].side == "river":
                            search_queue.append((next_col, next_row))
                    
                    move_idx += 1
                
                
                if grid[curr_row][curr_col] and grid[curr_row][curr_col].side == "river":
                    if grid[curr_row][curr_col].orientation == "horizontal":
                        # Check left
                        check_col = curr_col - 1
                        while check_col >= 0:
                            if (check_col, curr_row) in distance:
                                check_col -= 1
                                continue
                            if not grid[curr_row][check_col] or grid[curr_row][check_col].side == "river":
                                distance[(check_col, curr_row)] = distance[(curr_col, curr_row)] + 1
                                search_queue.append((check_col, curr_row))
                            else:
                                break
                            check_col -= 1
                        
                        
                        check_col = curr_col + 1
                        while check_col < width:
                            if (check_col, curr_row) in distance:
                                check_col += 1
                                continue
                            if not grid[curr_row][check_col] or grid[curr_row][check_col].side == "river":
                                distance[(check_col, curr_row)] = distance[(curr_col, curr_row)] + 1
                                search_queue.append((check_col, curr_row))
                            else:
                                break
                            check_col += 1
                    else:  
                        
                        check_row = curr_row - 1
                        while check_row >= 0:
                            if (curr_col, check_row) in distance:
                                check_row -= 1
                                continue
                            if not grid[check_row][curr_col] or grid[check_row][curr_col].side == "river":
                                distance[(curr_col, check_row)] = distance[(curr_col, curr_row)] + 1
                                search_queue.append((curr_col, check_row))
                            else:
                                break
                            check_row -= 1
                        
                        
                        check_row = curr_row + 1
                        while check_row < height:
                            if (curr_col, check_row) in distance:
                                check_row += 1
                                continue
                            if not grid[check_row][curr_col] or grid[check_row][curr_col].side == "river":
                                distance[(curr_col, check_row)] = distance[(curr_col, curr_row)] + 1
                                search_queue.append((curr_col, check_row))
                            else:
                                break
                            check_row += 1
    
    cost_list.sort()
    return cost_list




def assess_position(grid: List[List[Any]], player_name: str, 
                   height: int, width: int, goal_cols: List[int]) -> float:
    
    evaluation = 0.0
    enemy = switch_player(player_name)
    
    
    player_score = 0
    player_goal_row = circle_goal_line() if player_name == "circle" else square_goal_line(height)
    
    col_idx = 0
    while col_idx < len(goal_cols):
        col = goal_cols[col_idx]
        if within_grid(col, player_goal_row, height, width):
            token = grid[player_goal_row][col]
            if token and token.owner == player_name:
                player_score += 1
                if token.side == "stone":
                    player_score += 4
        col_idx += 1
    
    
    enemy_score = 0
    enemy_goal_row = circle_goal_line() if enemy == "circle" else square_goal_line(height)
    
    col_idx = 0
    while col_idx < len(goal_cols):
        col = goal_cols[col_idx]
        if within_grid(col, enemy_goal_row, height, width):
            token = grid[enemy_goal_row][col]
            if token and token.owner == enemy:
                enemy_score += 1
                if token.side == "stone":
                    enemy_score += 4
        col_idx += 1
    
    evaluation += player_score * 50000000
    evaluation -= enemy_score * 50000000
    
    
    enemy_stone_count = 0
    for col in goal_cols:
        if within_grid(col, enemy_goal_row, height, width):
            token = grid[enemy_goal_row][col]
            if token and token.owner == enemy:
                enemy_stone_count += 1
    
    enemy_near_win = enemy_stone_count >= 2
    
    DEFENSIVE_TIME = initial_time - 5
    OFFENSIVE_TIME = initial_time - 10
    
    if remaining_time > OFFENSIVE_TIME and not enemy_near_win:
        
        threat_zone = range(circle_goal_line() + 1) if player_name != "circle" else range(square_goal_line(height), height)
        threatened_cells = set()
        threat_value = 0
        
        col = 0
        while col < width:
            for row in threat_zone:
                token = grid[row][col]
                if token and token.owner != player_name and token.side == "river":
                    reachable = calculate_river_paths(grid, col, row, -1, -1, 
                                                    switch_player(player_name), 
                                                    height, width, goal_cols)
                    for cell in reachable:
                        if cell[1] in threat_zone:
                            threatened_cells.add(cell)
            col += 1
        
        for threatened in threatened_cells:
            if check_player_goal(threatened[0], threatened[1], player_name, height, width, goal_cols):
                threat_value += 10
            else:
                threat_value += 1
        
        evaluation -= threat_value * 50000000
        
        if remaining_time < DEFENSIVE_TIME:
            
            behind_value = 0
            if player_name != "square":
                goal_row = circle_goal_line()
                row = 0
                while row < goal_row:
                    col = 0
                    while col < width:
                        token = grid[row][col]
                        if token and token.owner == player_name:
                            behind_value += row
                        col += 1
                    row += 1
            else:
                goal_row = square_goal_line(height)
                row = goal_row + 1
                while row < height:
                    col = 0
                    while col < width:
                        token = grid[row][col]
                        if token and token.owner == player_name:
                            behind_value += (height - 1 - row)
                        col += 1
                    row += 1
            
            evaluation += behind_value * 200000
        
        max_search_depth = height + width
        player_paths = pathfinding_to_goal(grid, player_name, height, width, goal_cols, max_search_depth)
        enemy_paths = pathfinding_to_goal(grid, enemy, height, width, goal_cols, max_search_depth)
        
        weight_base = 500
        weight_delta = -5
        
        idx = 0
        while idx < min(7, len(player_paths)):
            evaluation -= player_paths[idx] * (weight_base + idx * weight_delta)
            idx += 1
        
        idx = 0
        while idx < min(7, len(enemy_paths)):
            attack_factor = 7
            evaluation += enemy_paths[idx] * (weight_base + idx * weight_delta) / attack_factor
            idx += 1
    else:
        
        behind_value = 0
        if player_name != "square":
            goal_row = circle_goal_line()
            row = 0
            while row < goal_row:
                col = 0
                while col < width:
                    token = grid[row][col]
                    if token and token.owner == player_name:
                        behind_value += row
                    col += 1
                row += 1
        else:
            goal_row = square_goal_line(height)
            row = goal_row + 1
            while row < height:
                col = 0
                while col < width:
                    token = grid[row][col]
                    if token and token.owner == player_name:
                        behind_value += (height - 1 - row)
                    col += 1
                row += 1
        
        evaluation += behind_value * 200000
        
        dist_weight = 500
        dist_decay = -10
        
        player_dists = closest_reachable_distance(grid, player_name, height, width, goal_cols)
        enemy_dists = closest_reachable_distance(grid, enemy, height, width, goal_cols)
        
        idx = 0
        while idx < len(player_dists):
            evaluation -= player_dists[idx] * max(0, (dist_weight + idx * dist_decay))
            idx += 1
        
        idx = 0
        while idx < len(enemy_dists):
            evaluation += enemy_dists[idx] * max(0, (dist_weight + idx * dist_decay))
            idx += 1
        
        total_p_dist, total_e_dist = aggregate_distances(grid, player_name, height, width, goal_cols)
        evaluation -= total_p_dist * 150
        evaluation += total_e_dist * 75
    
    
    outside_goal_pieces = 0
    
    col = 0
    while col < width:
        if col not in goal_cols and within_grid(col, player_goal_row, height, width):
            token = grid[player_goal_row][col]
            if token and token.owner == player_name:
                outside_goal_pieces += 1
        col += 1
    
    evaluation -= outside_goal_pieces * 4000
    
    
    enemy_river_total = 0
    row = 0
    while row < height:
        col = 0
        while col < width:
            token = grid[row][col]
            if token and token.owner == enemy and token.side == "river":
                enemy_river_total += 1
            col += 1
        row += 1
    
    evaluation -= enemy_river_total * 10000
    
    
    defense_value = 0
    opponent_goal_row = circle_goal_line() if player_name == "square" else square_goal_line(height)
    piece_count = 0
    
    for goal_col in goal_cols:
        
        adjacents = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        adj_idx = 0
        
        while adj_idx < len(adjacents):
            dc, dr = adjacents[adj_idx]
            check_col = goal_col + dc
            check_row = opponent_goal_row + dr
            
            if within_grid(check_col, check_row, height, width):
                token = grid[check_row][check_col]
                
                if token and token.owner == player_name:
                    if token.side != "river":
                        if piece_count < 8:
                            defense_value += 0.5
                        else:
                            defense_value -= 10
                        piece_count += 1
                    else:
                        if piece_count < 8:
                            if (token.orientation == "vertical" and check_row == opponent_goal_row) or \
                               (token.orientation == "horizontal" and check_row != opponent_goal_row):
                                defense_value += 2
                                piece_count += 1
                            else:
                                defense_value -= 3
                        else:
                            defense_value -= 10
                
                elif token and token.owner != player_name:
                    defense_value -= 2
            
            adj_idx += 1
    
    evaluation += defense_value * 20000000
    
    return evaluation
class FirstAgent:
    
    
    def __init__(self, side: str):
        self.player = side
        self.opponent = switch_player(side)
    
    
    
    def search_best_move(self, grid, height, width, goal_cols,
alpha_val, beta_val, current_depth, max_depth, is_maximizing):
        
        if current_depth == max_depth:
            return (assess_position(grid, self.player, height, width, goal_cols), {})

        
        if is_maximizing == 0:  # Our turn
                actions = create_all_actions(grid, self.player, height, width, goal_cols)
                best_score = -float("inf")
                selected_action = {}
                
                action_idx = 0
                while action_idx < len(actions):
                    if alpha_val >= beta_val:
                        break
                    
                    action = actions[action_idx]
                    valid, new_grid = apply_action(grid, action, self.player, height, width, goal_cols)
                    
                    if valid:
                        score, _ = self.search_best_move(new_grid, height, width, goal_cols,
                                                        alpha_val, beta_val, 
                                                        current_depth + 1, max_depth, 
                                                        is_maximizing ^ 1)
                        
                        if score > best_score:
                            # Add small randomness to avoid predictability
                            if random.random() <= 0.9:
                                best_score = score
                                selected_action = action
                        
                        alpha_val = max(alpha_val, best_score)
                    
                    action_idx += 1
                
                return (best_score, selected_action)
            
        else:  
                actions = create_all_actions(grid, self.opponent, height, width, goal_cols)
                worst_score = float("inf")
                selected_action = {}
                
                action_idx = 0
                while action_idx < len(actions):
                    if alpha_val >= beta_val:
                        break
                    
                    action = actions[action_idx]
                    valid, new_grid = apply_action(grid, action, self.opponent, height, width, goal_cols)
                    
                    if valid:
                        score, _ = self.search_best_move(new_grid, height, width, goal_cols,
                                                        alpha_val, beta_val, 
                                                        current_depth + 1, max_depth, 
                                                        is_maximizing ^ 1)
                        
                        if score < worst_score:
                            
                            if random.random() <= 0.9:
                                worst_score = score
                                selected_action = action
                        
                        beta_val = min(beta_val, worst_score)
                    
                    action_idx += 1
                
                return (worst_score, selected_action)

    def choose(self, grid: List[List[Any]], height: int, width: int, 
                goal_cols: List[int], player_time: float, enemy_time: float) -> Optional[Dict[str, Any]]:
            
            global remaining_time
            global initial_time
            
            if initial_time < 0:
                initial_time = player_time
            
            search_depth = 1
            
            time_start = time.perf_counter()
            _, chosen_action = self.search_best_move(grid, height, width, goal_cols,
                                                    -float("inf"), float("inf"), 
                                                    0, search_depth, 0)
            time_end = time.perf_counter()
            
            remaining_time = player_time - (time_end - time_start)
            
            return chosen_action

class Position(NamedTuple):
    
    x: int
    y: int

class GamePiece:
    
    def __init__(self, piece_owner: str, piece_type: str = "stone", flow_direction: Optional[str] = None):
        self.owner = piece_owner
        self.side = piece_type
        self.orientation = flow_direction if piece_type == "river" else None
        self._hash = hash((piece_owner, piece_type, flow_direction))
    
    def __hash__(self):
        return self._hash

class Move(NamedTuple):
    
    action: str
    source: Position
    target: Position
    auxiliary: Optional[Position] = None
    orientation: Optional[str] = None
    river_push: bool = False
    
    def to_dict(self):
        
        result = {
            "action": self.action,
            "from": [self.source.x, self.source.y]
        }
        if self.action in ["move", "push"]:
            result["to"] = [self.target.x, self.target.y]
        if self.action == "push" and self.auxiliary:
            result["pushed_to"] = [self.auxiliary.x, self.auxiliary.y]
            
            if self.river_push:
                result["river_push"] = True
        if self.orientation:
            result["orientation"] = self.orientation
        return result


class MoveGenerator:
    
    
    DIRECTIONS = [Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0)]
    
    def __init__(self, board_height: int, board_width: int, valid_columns: List[int]):
        self.height = board_height
        self.width = board_width
        self.valid_cols = frozenset(valid_columns)
        self._cache = {}
    
    def generate_all_moves(self, game_state: List[List[Any]], player: str) -> List[Move]:
        
        move_set = set()
        player_pieces = []
        
        
        for y in range(self.height):
            for x in range(self.width):
                piece = game_state[y][x]
                if piece and piece.owner == player:
                    player_pieces.append((Position(x, y), piece))
        
        
        for pos, piece in player_pieces:
            
            move_set.update(self._generate_basic_moves(game_state, pos, piece, player))
            
            
            move_set.update(self._generate_transformations(pos, piece))
            
            
            move_set.update(self._generate_river_moves(game_state, pos, player))
        
        return [move.to_dict() for move in move_set]
    
    def _generate_basic_moves(self, game_state: List[List[Any]], pos: Position, piece: GamePiece, player: str) -> Set[Move]:
        
        moves = set()
        
        for direction in self.DIRECTIONS:
            new_pos = Position(pos.x + direction.x, pos.y + direction.y)
            
            if not self._is_valid_position(new_pos) or self._is_opponent_score_area(new_pos, player):
                continue
            
            target = game_state[new_pos.y][new_pos.x]
            
            if target is None:
                moves.add(Move("move", pos, new_pos))
            elif piece.side == "stone":
                
                push_pos = Position(new_pos.x + direction.x, new_pos.y + direction.y)
                if (self._is_valid_position(push_pos) and 
                    game_state[push_pos.y][push_pos.x] is None and
                    not self._is_opponent_score_area(push_pos, player)):
                    moves.add(Move("push", pos, new_pos, push_pos))
            elif piece.side == "river" and target and target.side == "stone":
                
                if piece.orientation == "horizontal":
                    flow_directions = [(1, 0), (-1, 0)]
                else:  
                    flow_directions = [(0, 1), (0, -1)]
                
                
                for fd_x, fd_y in flow_directions:
                    if direction.x == fd_x and direction.y == fd_y:
                        
                        push_x = new_pos.x + fd_x
                        push_y = new_pos.y + fd_y
                        
                        while (self._is_valid_position(Position(push_x, push_y)) and
                            game_state[push_y][push_x] is None and
                            not self._is_opponent_score_area(Position(push_x, push_y), target.owner)):
                            moves.add(Move("push", pos, new_pos, Position(push_x, push_y), river_push=True))
                            push_x += fd_x
                            push_y += fd_y
                        break
        
        return moves
    
    def _generate_transformations(self, pos: Position, piece: GamePiece) -> Set[Move]:
        
        transforms = set()
        
        if piece.side == "stone":
            transforms.add(Move("flip", pos, pos, orientation="horizontal"))
            transforms.add(Move("flip", pos, pos, orientation="vertical"))
        else:
            transforms.add(Move("flip", pos, pos))
            transforms.add(Move("rotate", pos, pos))
        
        return transforms
    
    def _generate_river_moves(self, game_state: List[List[Any]], start_pos: Position, player: str) -> Set[Move]:
        
        river_moves = set()
        
        for direction in self.DIRECTIONS:
            adj_pos = Position(start_pos.x + direction.x, start_pos.y + direction.y)
            
            if not self._is_valid_position(adj_pos):
                continue
                
            adj_piece = game_state[adj_pos.y][adj_pos.x]
            if not adj_piece or adj_piece.side != 'river':
                continue
            
            
            pq = [(0, adj_pos)]
            visited = {start_pos, adj_pos}
            
            while pq:
                _, current = heapq.heappop(pq)
                current_piece = game_state[current.y][current.x]
                
                if not current_piece or current_piece.side != 'river':
                    continue
                
                flow_dirs = self._get_flow_directions(current_piece.orientation)
                
                for flow_dir in flow_dirs:
                    next_pos = Position(current.x + flow_dir.x, current.y + flow_dir.y)
                    
                    while (self._is_valid_position(next_pos) and 
                           not self._is_opponent_score_area(next_pos, player)):
                        
                        if next_pos in visited:
                            next_pos = Position(next_pos.x + flow_dir.x, next_pos.y + flow_dir.y)
                            continue
                        
                        target = game_state[next_pos.y][next_pos.x]
                        
                        if target is None:
                            river_moves.add(Move("move", start_pos, next_pos))
                            visited.add(next_pos)
                        elif target.side == 'river' and next_pos not in visited:
                            priority = abs(next_pos.x - start_pos.x) + abs(next_pos.y - start_pos.y)
                            heapq.heappush(pq, (priority, next_pos))
                            visited.add(next_pos)
                            break
                        else:
                            break
                        
                        next_pos = Position(next_pos.x + flow_dir.x, next_pos.y + flow_dir.y)
        
        return river_moves
    
    def _get_flow_directions(self, orientation: str) -> List[Position]:
        
        if orientation == 'vertical':
            return [Position(0, 1), Position(0, -1)]
        return [Position(1, 0), Position(-1, 0)]
    
    def _is_valid_position(self, pos: Position) -> bool:
        
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def _is_opponent_score_area(self, pos: Position, player: str) -> bool:
        
        if player == "circle":
            return pos.y == self.height - 3 and pos.x in self.valid_cols
        return pos.y == 2 and pos.x in self.valid_cols




class SecondAgent:

    
    
    def __init__(self, player_type: str):
        self.player = player_type
        self.opponent = "square" if player_type == "circle" else "circle"
        self.board_info = {}
        self.move_generator = None
        self.evaluation_cache = {}
        self.transposition_table = {}
        
    def choose(self, game_state: List[List[Any]], board_height: int, board_width: int, 
               valid_columns: List[int], time_self: float, time_rival: float) -> Optional[Dict[str, Any]]:
        
        
        self.board_info = {
            'height': board_height,
            'width': board_width,
            'valid_cols': frozenset(valid_columns),
            'player_goal': 2 if self.player == "circle" else board_height - 3,
            'opponent_goal': 2 if self.opponent == "circle" else board_height - 3
        }
        
        self.move_generator = MoveGenerator(board_height, board_width, valid_columns)
        
        
        available_moves = self.move_generator.generate_all_moves(game_state, self.player)
        if not available_moves:
            return None
        
        
        for move in available_moves:
            if self._is_winning_move(game_state, move):
                return move
        
        
        threat_level = self._analyze_threats(game_state)
        
        if threat_level == 0: 
            return self._minimax_search(game_state, available_moves, 2)
        else:  
            return self._defensive_strategy(game_state, available_moves)
    
    def _minimax_search(self, game_state: List[List[Any]], moves: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
        
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        
        move_heap = []
        for move in moves:
            priority = -self._quick_evaluate_move(game_state, move)
            heapq.heappush(move_heap, (priority, random.random(), move))
        
        while move_heap:
            _, _, move = heapq.heappop(move_heap)
            
            new_state = self._apply_move_optimized(game_state, move, self.player)
            if not new_state:
                continue
            
            value = self._minimax(new_state, depth - 1, alpha, beta, False)
            
            if value > alpha:
                alpha = value
                best_move = move
        
        return best_move or random.choice(moves)
    
    def _minimax(self, state: List[List[Any]], depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        
        state_hash = self._hash_state(state)
        
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]
        
        if depth == 0:
            value = self._evaluate_state(state)
            self.transposition_table[state_hash] = value
            return value
        
        current_player = self.player if maximizing else self.opponent
        moves = self.move_generator.generate_all_moves(state, current_player)
        
        if not moves:
            value = self._evaluate_state(state)
            self.transposition_table[state_hash] = value
            return value
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves[:10]:  
                new_state = self._apply_move_optimized(state, move, current_player)
                if new_state:
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            self.transposition_table[state_hash] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves[:10]:
                new_state = self._apply_move_optimized(state, move, current_player)
                if new_state:
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            self.transposition_table[state_hash] = min_eval
            return min_eval
    
    def _analyze_threats(self, game_state: List[List[Any]]) -> int:
        
        opp_goal = self.board_info['opponent_goal']
        opp_stones = 0
        threatening_pieces = []
        
        
        for col in self.board_info['valid_cols']:
            piece = game_state[opp_goal][col]
            if piece and piece.owner == self.opponent and piece.side == 'stone':
                opp_stones += 1
        
        
        if opp_stones == 3:
            opp_moves = self.move_generator.generate_all_moves(game_state, self.opponent)
            for move in opp_moves:
                if move['action'] == 'move':
                    dest = Position(move['to'][0], move['to'][1])
                    if dest.y == opp_goal and dest.x in self.board_info['valid_cols']:
                        src = Position(move['from'][0], move['from'][1])
                        if game_state[src.y][src.x].side == 'stone':
                            return 2  # Critical threat
        
        
        for y in range(self.board_info['height']):
            for x in range(self.board_info['width']):
                piece = game_state[y][x]
                if piece and piece.owner == self.opponent and piece.side == 'stone':
                    distance = abs(y - opp_goal)
                    if distance <= 2:
                        threatening_pieces.append((Position(x, y), distance))
        
        
        if len(threatening_pieces) >= 3 and opp_stones >= 2:
            return 2
        elif len(threatening_pieces) >= 2 or opp_stones >= 2:
            return 1
        return 0
    
    def _defensive_strategy(self, game_state: List[List[Any]], moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        defensive_candidates = []
        
        
        for move in moves:
            defense_score = self._calculate_defensive_value(game_state, move)
            if defense_score > 0:
                defensive_candidates.append((defense_score, move))
        
        if defensive_candidates:
            
            defensive_candidates.sort(key=lambda x: -x[0])
            return defensive_candidates[0][1]
        
        
        return self._minimax_search(game_state, moves, 1)
    
    def _calculate_defensive_value(self, game_state: List[List[Any]], move: Dict[str, Any]) -> float:
        
        score = 0.0
        
       
        test_state = self._apply_move_optimized(game_state, move, self.player)
        if not test_state:
            return 0.0
        
        
        opp_moves = self.move_generator.generate_all_moves(test_state, self.opponent)
        opp_goal = self.board_info['opponent_goal']
        
        blocked_wins = 0
        for opp_move in opp_moves:
            if opp_move['action'] == 'move':
                dest = Position(opp_move['to'][0], opp_move['to'][1])
                if dest.y == opp_goal and dest.x in self.board_info['valid_cols']:
                    src = Position(opp_move['from'][0], opp_move['from'][1])
                    if test_state[src.y][src.x] and test_state[src.y][src.x].side == 'stone':
                        
                        blocked_wins += 1
        
        score += blocked_wins * 10000
        
        
        if move['action'] == 'move':
            dest = Position(move['to'][0], move['to'][1])
            
            if abs(dest.y - opp_goal) <= 1:
                score += 500
            
            if dest.x in self.board_info['valid_cols']:
                score += 300
        
        
        elif move['action'] == 'push':
            pushed_to = Position(move['pushed_to'][0], move['pushed_to'][1])
            original = Position(move['to'][0], move['to'][1])
            piece = game_state[original.y][original.x]
            
            if piece and piece.owner == self.opponent:
                dist_before = abs(original.y - opp_goal)
                dist_after = abs(pushed_to.y - opp_goal)
                score += (dist_after - dist_before) * 1000
        
        return score
    
    def _is_winning_move(self, game_state: List[List[Any]], move: Dict[str, Any]) -> bool:
        
        if move['action'] != 'move':
            return False
        
        src = Position(move['from'][0], move['from'][1])
        dest = Position(move['to'][0], move['to'][1])
        piece = game_state[src.y][src.x]
        
        if piece.side != 'stone':
            return False
        
        player_goal = self.board_info['player_goal']
        if dest.y == player_goal and dest.x in self.board_info['valid_cols']:
            
            stone_count = sum(1 for col in self.board_info['valid_cols']
                            if game_state[player_goal][col] and
                            game_state[player_goal][col].owner == self.player and
                            game_state[player_goal][col].side == 'stone')
            return stone_count == 3
        
        return False
    
    def _evaluate_state(self, state: List[List[Any]]) -> float:
        
        score = 0.0
        player_goal = self.board_info['player_goal']
        opp_goal = self.board_info['opponent_goal']
        
        
        for y in range(self.board_info['height']):
            for x in range(self.board_info['width']):
                piece = state[y][x]
                if not piece:
                    continue
                
                if piece.owner == self.player:
                    
                    if piece.side == 'stone' and y == player_goal and x in self.board_info['valid_cols']:
                        score += 5000
                    
                    
                    elif piece.side == 'stone':
                        distance = abs(y - player_goal)
                        score += (10 - distance) * 50
                        
                        # Column preference
                        if x in self.board_info['valid_cols']:
                            score += 100
                    
                    
                    elif piece.side == 'river':
                        if piece.orientation == 'vertical' and abs(y - player_goal) > 1:
                            score += 200
                        elif piece.orientation == 'horizontal':
                            score += 100
                
                else:  
                   
                    if piece.side == 'stone' and y == opp_goal and x in self.board_info['valid_cols']:
                        score -= 5000
                    
                    
                    elif piece.side == 'stone':
                        distance = abs(y - opp_goal)
                        score -= (10 - distance) * 50
        
        
        player_moves = len(self.move_generator.generate_all_moves(state, self.player))
        opp_moves = len(self.move_generator.generate_all_moves(state, self.opponent))
        score += (player_moves - opp_moves) * 5
        
        return score
    
    def _quick_evaluate_move(self, game_state: List[List[Any]], move: Dict[str, Any]) -> float:
        
        score = 0.0
        action = move['action']
        
        if action == 'move':
            src = Position(move['from'][0], move['from'][1])
            dest = Position(move['to'][0], move['to'][1])
            piece = game_state[src.y][src.x]
            
            if piece.side == 'stone':
                
                if dest.y == self.board_info['player_goal'] and dest.x in self.board_info['valid_cols']:
                    score += 1000
                
                
                progress = abs(src.y - self.board_info['player_goal']) - abs(dest.y - self.board_info['player_goal'])
                score += progress * 100
        
        elif action == 'flip':
            src = Position(move['from'][0], move['from'][1])
            piece = game_state[src.y][src.x]
            
            
            if piece.side == 'river' and src.y == self.board_info['player_goal'] and src.x in self.board_info['valid_cols']:
                score += 800
        
        elif action == 'push':
            
            to_pos = Position(move['to'][0], move['to'][1])
            pushed_piece = game_state[to_pos.y][to_pos.x]
            if pushed_piece and pushed_piece.owner == self.opponent:
                score += 300
        
        return score
    
    def _apply_move_optimized(self, game_state: List[List[Any]], move: Dict[str, Any], player: str) -> Optional[List[List[Any]]]:
        
        new_state = [row[:] for row in game_state]
        
        try:
            action = move['action']
            src = Position(move['from'][0], move['from'][1])
            
            piece = new_state[src.y][src.x]
            if not piece or piece.owner != player:
                return None
            
            if action == 'move':
                dest = Position(move['to'][0], move['to'][1])
                
                if new_state[dest.y][dest.x] is None:
                    new_state[dest.y][dest.x] = piece
                    new_state[src.y][src.x] = None
                    return new_state
                
                
                pushed_to = move.get('pushed_to')
                if pushed_to:
                    push_pos = Position(pushed_to[0], pushed_to[1])
                    if new_state[push_pos.y][push_pos.x] is None:
                        new_state[push_pos.y][push_pos.x] = new_state[dest.y][dest.x]
                        new_state[dest.y][dest.x] = piece
                        new_state[src.y][src.x] = None
                        return new_state
            
            elif action == 'push':
                dest = Position(move['to'][0], move['to'][1])
                push_pos = Position(move['pushed_to'][0], move['pushed_to'][1])
                
                if new_state[dest.y][dest.x] and new_state[push_pos.y][push_pos.x] is None:
                    
                    original_piece = piece
                    
                    new_state[push_pos.y][push_pos.x] = new_state[dest.y][dest.x]
                    new_state[dest.y][dest.x] = piece
                    new_state[src.y][src.x] = None
                    
                    
                    if move.get('river_push', False) and original_piece.side == 'river':
                        new_state[dest.y][dest.x] = GamePiece(original_piece.owner, 'stone', None)
                    
                    return new_state
            
            elif action == 'flip':
                
                if piece.side == 'stone':
                    new_piece = GamePiece(piece.owner, 'river', move.get('orientation', 'horizontal'))
                else:
                    new_piece = GamePiece(piece.owner, 'stone', None)
                new_state[src.y][src.x] = new_piece
                return new_state
            
            elif action == 'rotate':
                if piece.side == 'river':
                    new_orientation = 'vertical' if piece.orientation == 'horizontal' else 'horizontal'
                    new_piece = GamePiece(piece.owner, 'river', new_orientation)
                    new_state[src.y][src.x] = new_piece
                    return new_state
        
        except Exception:
            pass
        
        return None
    
    def _hash_state(self, state: List[List[Any]]) -> int:
        
        hash_value = 0
        for y in range(self.board_info['height']):
            for x in range(self.board_info['width']):
                piece = state[y][x]
                if piece:
                    
                    piece_hash = hash((x, y, piece.owner, piece.side, piece.orientation))
                    hash_value ^= piece_hash
        return hash_value
class StudentAgent:
    
    
    def __init__(self, side: str):
        self.player = side
        self.opponent = switch_player(side)
        self.first_agent = FirstAgent(side)
        self.second_agent = SecondAgent(side)
    
    def choose(self, grid: List[List[Any]], height: int, width: int, 
               goal_cols: List[int], player_time: float, enemy_time: float) -> Optional[Dict[str, Any]]:
        
        global turn_count
        turn_count += 1
        
        
        if turn_count <=125:
            
            return self.first_agent.choose(grid, height, width, goal_cols, player_time, enemy_time)
        else:
            
            
            return self.second_agent.choose(grid, height, width, goal_cols, player_time, enemy_time)
    
    def _convert_grid_format(self, grid: List[List[Any]], height: int, width: int) -> List[List[Any]]:
        
        new_grid = [[None for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                if grid[y][x] is not None:
                    old_piece = grid[y][x]
                    # Convert GameToken to GamePiece
                    if hasattr(old_piece, 'owner'):
                        new_grid[y][x] = GamePiece(
                            old_piece.owner,
                            old_piece.side,
                            old_piece.orientation if hasattr(old_piece, 'orientation') else None
                        )
        
        return new_grid