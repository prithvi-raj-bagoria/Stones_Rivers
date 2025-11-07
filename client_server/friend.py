"""
Student Agent Implementation for River and Stones Game

ARCHITECTURE OVERVIEW:
=====================
MINIMAX SEARCH + STRATEGIC HEURISTICS

This agent combines game tree search with domain-specific heuristics:

LAYER 1-7: STRATEGIC EVALUATION (Heuristic Foundation)
   1. Board Evaluation: Score current position
   2. Strategic Mode: Determine game phase (attacking/defending)
   3. Targeted Move Generation: Generate relevant moves only
   4. Move Categorization: Classify by purpose (winning/scoring/attacking)
   5. Move Scoring: Assign numerical values
   6. Threat Assessment: Identify opponent dangers
   7. Position Evaluation: Complete board analysis

LAYER 8-11: MINIMAX SEARCH (Look-Ahead)
   8. Zobrist Hashing: Fast board position comparison (O(1))
   9. Transposition Table: Cache evaluated positions
   10. Move Ordering: Search best moves first (improves pruning)
   11. Minimax + Alpha-Beta: Game tree search with pruning

HOW IT WORKS:
-------------
1. **Strategic Assessment**: Quick analysis determines game state
   - "Can we win?" → Take winning move immediately
   - "Is opponent threatening?" → Focus on defense
   - "Are we ahead?" → Play safe, protect lead

2. **Minimax Search**: Look ahead to predict opponent responses
   - Search depth 1-3 (adaptive based on time)
   - Alpha-beta pruning eliminates bad branches
   - Transposition table avoids re-computing positions
   - Move ordering: search promising moves first

3. **Evaluation Function**: Scores board positions
   - Stones in scoring area: +1000 per stone
   - Stone advancement: +100 per stone closer to goal
   - Opponent's progress: negative values
   - Returns single number: positive = good, negative = bad

4. **Time Management**: Adaptive search depth
   - More time (>30s) → Search depth 3 (our move + opponent + our response)
   - Medium time (10-30s) → Search depth 2 (our move + opponent)
   - Low time (<10s) → Search depth 1 or fallback to heuristic

WHY MINIMAX + HEURISTICS:
-------------------------
- **Minimax**: Looks ahead, predicts opponent responses
- **Heuristics**: Guides search, prunes bad branches early
- **Combined**: Smart search + domain knowledge = Strong play
- **No pure brute force**: Would be too slow without heuristics

EXAMPLE DECISION FLOW:
---------------------
```
1. Board State: Opponent has 3 stones in scoring area
2. Strategic Mode: DESPERATE_DEFENSE (opponent about to win!)
3. Move Generation: Generate ONLY defensive moves (30 moves vs 100+)
4. Minimax Search:
   - Try defensive move A:
     * Opponent's best response: Score opponent stones
     * Resulting position: -10000 (we lose)
   - Try defensive move B:
     * Opponent's best response: Advance stone
     * Resulting position: -200 (we survive)
   - Alpha-beta: Prune remaining moves (B is good enough)
5. Decision: Play move B (prevents opponent win)
```

MODULAR DESIGN:
--------------
Each layer is independent and testable:
- Want better evaluation? → Modify evaluate_position()
- Want deeper search? → Increase search_depth
- Want faster search? → Improve move ordering
- Want to test heuristics? → Disable minimax, use fallback

KEY OPTIMIZATIONS:
-----------------
1. **Zobrist Hashing**: O(1) position lookup vs O(n²) comparison
2. **Alpha-Beta Pruning**: 50-90% fewer nodes searched
3. **Transposition Table**: Avoid recomputing same positions
4. **Move Ordering**: Search best moves first (more cutoffs)
5. **Targeted Generation**: Generate 30 relevant moves vs 100+ total
6. **Adaptive Depth**: Search deeper when time allows
"""

import random
import copy
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"

# ==================== GAME CONSTANTS (DERIVED FROM RULES) ====================
# ALL weights derived from game rules - NO ARBITRARY NUMBERS!

class GameConstants:
    """
    All game constants and derived weights.
    Everything is based on game rules, not arbitrary numbers.
    """
    
    def __init__(self, rows: int, cols: int):
        # Core game rules
        self.STONES_TO_WIN = 4
        self.rows = rows
        self.cols = cols
        
        # Derived constants
        self.MAX_DISTANCE = rows + cols  # Theoretical max Manhattan distance
        self.SCORING_AREA_SIZE = 4  # Width of scoring area
        
        # ===== NORMALIZED WEIGHTS (all relative to STONES_TO_WIN) =====
        # Philosophy: 1 stone in scoring area = 1.0 progress toward victory
        
        # Material evaluation (stones in scoring area)
        self.WEIGHT_SCORING_STONE = 1.0 / self.STONES_TO_WIN  # Each stone = 25% toward win
        
        # Distance evaluation (advancement toward goal)
        self.WEIGHT_DISTANCE = self.WEIGHT_SCORING_STONE * 0.6  # 60% as important as scored stone
        
        # Threat evaluation (pieces close to scoring)
        self.WEIGHT_IMMEDIATE_THREAT = self.WEIGHT_SCORING_STONE * 0.5  # 1 move away
        self.WEIGHT_NEAR_THREAT = self.WEIGHT_SCORING_STONE * 0.25  # 2-3 moves away
        
        # River network evaluation (connectivity to goal)
        self.WEIGHT_RIVER_HIGHWAY = self.WEIGHT_SCORING_STONE * 0.4  # Connected path
        
        # Push opportunities
        self.WEIGHT_PUSH_THREAT = self.WEIGHT_SCORING_STONE * 0.3
        
        # Lane control (4 scoring columns)
        self.WEIGHT_LANE_CONTROL = self.WEIGHT_SCORING_STONE * 0.2
        
        # Mobility
        self.WEIGHT_MOBILITY = self.WEIGHT_SCORING_STONE * 0.15
        
        # Defensive barriers
        self.WEIGHT_BARRIER = self.WEIGHT_SCORING_STONE * 0.1
        
        # ===== DISTANCE THRESHOLDS (relative to board size) =====
        # "Close to goal" = within 1/3 of board height
        self.THREAT_DISTANCE_IMMEDIATE = max(1, rows // 6)  # 1-2 moves away
        self.THREAT_DISTANCE_NEAR = max(2, rows // 3)  # 3-4 moves away
        
        # ===== STRATEGIC MODE THRESHOLDS (relative to STONES_TO_WIN) =====
        # "Big advantage" = 50% of stones needed to win
        self.STRATEGIC_ADVANTAGE_BIG = self.STONES_TO_WIN // 2  # 2 stones
        # "Small advantage" = 25% of stones needed to win  
        self.STRATEGIC_ADVANTAGE_SMALL = max(1, self.STONES_TO_WIN // 4)  # 1 stone

# ==================== LAYER 1: MOVE GENERATION ====================
# This layer generates ALL possible legal moves without evaluating them

def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all valid moves for a specific piece.
    
    Returns moves categorized by action type:
    - "move": Simple movement to adjacent empty cell
    - "push": Push opponent's piece
    - "flip": Change stone to river or vice versa
    - "rotate": Change river orientation
    """
    moves = []
    piece = board[y][x]
    
    if piece is None or piece.owner != player:
        return moves
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    if piece.side == "stone":
        # STONE ACTIONS
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            
            if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                continue
            
            if board[ny][nx] is None:
                # Simple move to empty cell
                moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
            elif board[ny][nx].owner != player:
                # Push opponent's piece (stones can only push stones)
                target_piece = board[ny][nx]
                pushed_player = target_piece.owner  # The piece being pushed
                px, py = nx + dx, ny + dy
                
                # CRITICAL: Check if pushed_to is opponent scoring area from PUSHED player's perspective
                if (in_bounds(px, py, rows, cols) and 
                    board[py][px] is None and 
                    not is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols)):
                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
        
        # Stone can flip to river (choose orientation)
        for orientation in ["horizontal", "vertical"]:
            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
    
    else:  # RIVER PIECE
        # River can flip back to stone
        moves.append({"action": "flip", "from": [x, y]})
        
        # River can rotate orientation
        moves.append({"action": "rotate", "from": [x, y]})
    
    return moves

def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    LAYER 1: Generate ALL legal moves for the current player.
    
    This is the first step - we gather every possible action without filtering.
    Later layers will evaluate and rank these moves.
    """
    all_moves = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player:
                piece_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols)
                all_moves.extend(piece_moves)
    
    return all_moves


# ==================== LAYER 2: MOVE CATEGORIZATION ====================
# This layer categorizes moves by their strategic purpose

def categorize_move(move: Dict[str, Any], board: List[List[Any]], player: str, 
                    rows: int, cols: int, score_cols: List[int]) -> str:
    """
    Categorize a move by its strategic purpose.
    
    Categories:
    - "WINNING": Move that wins the game (4th stone to scoring area)
    - "SCORING": Move that places a stone in our scoring area
    - "ADVANCING": Move that gets our stones closer to scoring
    - "ATTACKING": Move that disrupts opponent
    - "DEFENSIVE": Move that protects our position
    - "SETUP": Move that prepares for future plays (flips, rotates)
    """
    action = move.get("action")
    from_pos = move.get("from")
    to_pos = move.get("to")
    
    # Check if this is a scoring move
    if action == "move" and to_pos:
        tx, ty = to_pos
        if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
            # Check if this would be our 4th stone (winning move)
            current_score = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
            if current_score == 3:
                return "WINNING"
            return "SCORING"
    
    # Check if this is an attacking move (push opponent)
    if action == "push":
        return "ATTACKING"
    
    # Check if moving forward (advancing toward goal)
    if action == "move" and to_pos and from_pos:
        fx, fy = from_pos
        tx, ty = to_pos
        
        # Check if we're moving toward our scoring row
        if player == "circle":
            if ty < fy:  # Moving up toward circle's scoring area
                return "ADVANCING"
        else:  # square
            if ty > fy:  # Moving down toward square's scoring area
                return "ADVANCING"
    
    # Flips and rotates are setup moves
    if action in ["flip", "rotate"]:
        return "SETUP"
    
    # Everything else is defensive
    return "DEFENSIVE"


def filter_moves_by_category(moves: List[Dict[str, Any]], board: List[List[Any]], 
                             player: str, rows: int, cols: int, score_cols: List[int], 
                             preferred_categories: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    LAYER 2: Organize moves into categories for easier decision making.
    
    Returns:
        Dictionary mapping category names to lists of moves
    """
    categorized = {
        "WINNING": [],
        "SCORING": [],
        "ADVANCING": [],
        "ATTACKING": [],
        "DEFENSIVE": [],
        "SETUP": []
    }
    
    for move in moves:
        category = categorize_move(move, board, player, rows, cols, score_cols)
        categorized[category].append(move)
    
    return categorized


# ==================== LAYER 3: MOVE EVALUATION ====================
# This layer assigns scores to moves based on strategic value

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0
    
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    
    return count

def calculate_distance_to_scoring(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """
    Calculate Manhattan distance from a position to the closest scoring cell.
    Lower is better (closer to goal).
    """
    if player == "circle":
        goal_row = top_score_row()
    else:
        goal_row = bottom_score_row(rows)
    
    # Distance to goal row
    row_distance = abs(y - goal_row)
    
    # Distance to nearest scoring column
    col_distance = min(abs(x - sc) for sc in score_cols)
    
    return row_distance + col_distance

def evaluate_move_score(move: Dict[str, Any], board: List[List[Any]], player: str, 
                        rows: int, cols: int, score_cols: List[int]) -> float:
    """
    LAYER 3: Calculate a numerical score for a move.
    Higher scores = better moves.
    
    Scoring rubric:
    - Winning move: 10000 points
    - Scoring move: 1000 points
    - Advancing move: 100 - distance_to_goal
    - Attacking move: 50 points
    - Other moves: 10 points
    """
    category = categorize_move(move, board, player, rows, cols, score_cols)
    
    # Priority 1: Winning moves
    if category == "WINNING":
        return 10000.0
    
    # Priority 2: Scoring moves
    if category == "SCORING":
        return 1000.0
    
    # Priority 3: Advancing moves (score based on how close to goal)
    if category == "ADVANCING":
        to_pos = move.get("to")
        if to_pos:
            tx, ty = to_pos
            distance = calculate_distance_to_scoring(tx, ty, player, rows, cols, score_cols)
            # Closer to goal = higher score
            return 100.0 - distance
    
    # Priority 4: Attacking moves
    if category == "ATTACKING":
        return 50.0
    
    # Priority 5: Setup moves
    if category == "SETUP":
        return 20.0
    
    # Priority 6: Defensive moves
    return 10.0

def rank_moves_by_score(moves: List[Dict[str, Any]], board: List[List[Any]], 
                        player: str, rows: int, cols: int, score_cols: List[int]) -> List[Tuple[Dict[str, Any], float]]:
    """
    LAYER 3: Score all moves and return them sorted by score (best first).
    
    Returns:
        List of (move, score) tuples, sorted by score (descending)
    """
    scored_moves = []
    
    for move in moves:
        score = evaluate_move_score(move, board, player, rows, cols, score_cols)
        scored_moves.append((move, score))
    
    # Sort by score (highest first)
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    return scored_moves


# ==================== LAYER 4: POSITION EVALUATION ====================
# This layer evaluates overall board positions (for deeper analysis)

# ==================== TIER 1 EVALUATIONS (GAME-DECISIVE) ====================
# These evaluations determine immediate victory/defeat conditions

def evaluate_material_normalized(board: List[List[Any]], player: str, rows: int, cols: int, 
                                 score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 1: Evaluate material (stones in scoring area).
    Returns NORMALIZED score (0.0 to 1.0).
    
    This is THE MOST IMPORTANT evaluation - directly determines victory.
    
    Returns:
        {
            "my_stones": int - Count of my stones in scoring area,
            "opp_stones": int - Count of opponent stones,
            "my_progress": float - 0.0 to 1.0 (0 = no stones, 1.0 = won),
            "opp_progress": float - 0.0 to 1.0,
            "advantage": float - -1.0 to +1.0 (relative advantage)
        }
    """
    opponent = get_opponent(player)
    
    my_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opp_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    # Normalize to 0-1 range
    my_progress = min(1.0, my_stones / constants.STONES_TO_WIN)
    opp_progress = min(1.0, opp_stones / constants.STONES_TO_WIN)
    
    # Relative advantage (-1 to +1)
    advantage = (my_stones - opp_stones) / constants.STONES_TO_WIN
    
    return {
        "my_stones": my_stones,
        "opp_stones": opp_stones,
        "my_progress": my_progress,
        "opp_progress": opp_progress,
        "advantage": advantage
    }


def evaluate_distance_normalized(board: List[List[Any]], player: str, rows: int, cols: int,
                                  score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 1: Evaluate distance from scoring area (advancement).
    Returns NORMALIZED score.
    
    Measures how close pieces are to scoring. Closer = better position.
    
    Returns:
        {
            "my_avg_distance": float - Average distance (normalized 0-1),
            "opp_avg_distance": float,
            "my_closest": int - Distance of closest stone,
            "opp_closest": int,
            "advantage": float - Positive means we're closer
        }
    """
    opponent = get_opponent(player)
    
    my_distances = []
    opp_distances = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.side == "stone":
                distance = calculate_distance_to_scoring(x, y, piece.owner, rows, cols, score_cols)
                
                if piece.owner == player:
                    my_distances.append(distance)
                else:
                    opp_distances.append(distance)
    
    # Calculate averages
    my_avg = sum(my_distances) / len(my_distances) if my_distances else constants.MAX_DISTANCE
    opp_avg = sum(opp_distances) / len(opp_distances) if opp_distances else constants.MAX_DISTANCE
    
    # Normalize (invert so 0=far, 1=close)
    my_avg_norm = 1.0 - min(1.0, my_avg / constants.MAX_DISTANCE)
    opp_avg_norm = 1.0 - min(1.0, opp_avg / constants.MAX_DISTANCE)
    
    # Closest pieces
    my_closest = min(my_distances) if my_distances else constants.MAX_DISTANCE
    opp_closest = min(opp_distances) if opp_distances else constants.MAX_DISTANCE
    
    # Advantage: positive if we're closer
    advantage = opp_avg_norm - my_avg_norm
    
    return {
        "my_avg_distance": my_avg_norm,
        "opp_avg_distance": opp_avg_norm,
        "my_closest": my_closest,
        "opp_closest": opp_closest,
        "advantage": advantage
    }


def evaluate_can_win_this_turn(board: List[List[Any]], player: str, rows: int, cols: int,
                                score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 1: Check if player can win this turn.
    
    Returns:
        {
            "can_win": bool - Can score 4th stone this turn,
            "winning_moves_count": int - Number of ways to win,
            "winning_moves": List[Dict] - Actual winning moves found
        }
    """
    my_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    
    if my_stones != 3:
        return {"can_win": False, "winning_moves_count": 0, "winning_moves": []}
    
    # Find actual winning moves
    goal_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    winning_moves = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Check if adjacent to empty scoring cell
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        if ny == goal_row and nx in score_cols and board[ny][nx] is None:
                            winning_moves.append({
                                "action": "move",
                                "from": [x, y],
                                "to": [nx, ny]
                            })
    
    return {
        "can_win": len(winning_moves) > 0,
        "winning_moves_count": len(winning_moves),
        "winning_moves": winning_moves
    }


# ==================== TIER 2 EVALUATIONS (TACTICAL) ====================
# These evaluations measure tactical advantages

def evaluate_river_networks(board: List[List[Any]], player: str, rows: int, cols: int,
                            score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 2: Evaluate river network connectivity.
    
    Rivers create "highways" for rapid movement. Connected rivers are powerful.
    This enhanced version detects actual connectivity and paths to scoring area.
    
    Returns:
        {
            "my_river_count": int,
            "opp_river_count": int,
            "my_connectivity": float - 0.0 to 1.0 (how connected),
            "opp_connectivity": float,
            "my_highways_to_goal": int - Connected paths to scoring area,
            "opp_highways_to_goal": int,
            "advantage": float
        }
    """
    opponent = get_opponent(player)
    
    # Find all rivers
    my_rivers = []
    opp_rivers = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.side == "river":
                river_info = {
                    "x": x,
                    "y": y,
                    "orientation": piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                }
                if piece.owner == player:
                    my_rivers.append(river_info)
                else:
                    opp_rivers.append(river_info)
    
    # Helper function to check if two rivers are connected
    def are_rivers_connected(r1, r2):
        """Check if two rivers touch (share edge or corner)"""
        dx = abs(r1["x"] - r2["x"])
        dy = abs(r1["y"] - r2["y"])
        # Adjacent if Manhattan distance <= 2 (allows diagonal)
        return (dx <= 1 and dy <= 1) and not (dx == 0 and dy == 0)
    
    # Build connectivity graph using BFS to find connected components
    def count_connected_components(rivers):
        """Count connected river networks"""
        if not rivers:
            return 0.0, []  # Return empty list of components, not 0
        
        visited = set()
        components = []
        
        for i, river in enumerate(rivers):
            if i in visited:
                continue
            
            # BFS to find all rivers in this component
            component = []
            queue = [i]
            visited.add(i)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                
                # Check all other rivers for connections
                for j, other_river in enumerate(rivers):
                    if j not in visited and are_rivers_connected(rivers[current], other_river):
                        visited.add(j)
                        queue.append(j)
            
            components.append(component)
        
        # Connectivity = 1 - (num_components / num_rivers)
        # If all rivers connected = 1 component = connectivity 1.0
        # If all rivers isolated = num_rivers components = connectivity 0.0
        num_components = len(components)
        connectivity = 1.0 - ((num_components - 1) / max(1, len(rivers)))
        
        return connectivity, components
    
    # Calculate connectivity for both players
    my_connectivity, my_components = count_connected_components(my_rivers)
    opp_connectivity, opp_components = count_connected_components(opp_rivers)
    
    # Check for "highways to goal" - connected rivers reaching toward scoring area
    def count_highways_to_goal(rivers, components, goal_row):
        """Count river networks that create paths toward goal"""
        highways = 0
        
        for component in components:
            if len(component) < 2:  # Single river isn't a highway
                continue
            
            # Check if component has rivers close to goal
            component_rivers = [rivers[i] for i in component]
            min_distance = min(abs(r["y"] - goal_row) for r in component_rivers)
            
            # Highway if connected network reaches within 1/3 of board toward goal
            if min_distance <= rows // 3:
                highways += 1
        
        return highways
    
    my_goal_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    opp_goal_row = bottom_score_row(rows) if player == "circle" else top_score_row()
    
    my_highways = count_highways_to_goal(my_rivers, my_components, my_goal_row)
    opp_highways = count_highways_to_goal(opp_rivers, opp_components, opp_goal_row)
    
    # Advantage combines connectivity and highways
    advantage = (my_connectivity - opp_connectivity) * 0.5 + \
                (my_highways - opp_highways) / max(1, max(len(my_components), len(opp_components))) * 0.5
    
    return {
        "my_river_count": len(my_rivers),
        "opp_river_count": len(opp_rivers),
        "my_connectivity": my_connectivity,
        "opp_connectivity": opp_connectivity,
        "my_highways_to_goal": my_highways,
        "opp_highways_to_goal": opp_highways,
        "advantage": advantage
    }


def evaluate_push_opportunities(board: List[List[Any]], player: str, rows: int, cols: int,
                                score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 2: Evaluate push opportunities and vulnerabilities.
    
    Enhanced to include RIVER PUSHES which are much more powerful than stone pushes.
    River pushes can move opponent stones multiple spaces (5-10x more disruptive).
    
    Returns:
        {
            "my_stone_pushes": int - Stone-to-stone pushes (1 space),
            "my_river_pushes": int - River-to-stone pushes (multi-space),
            "my_river_push_distance": int - Total spaces opponent can be pushed by rivers,
            "my_vulnerabilities": int - My pieces opponent can push,
            "advantage": float - Normalized advantage
        }
    """
    opponent = get_opponent(player)
    
    my_stone_pushes = 0
    my_river_pushes = 0
    my_river_push_distance = 0
    my_vulnerabilities = 0
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue
            
            # Count stone-to-stone pushes
            if piece.owner == player and piece.side == "stone":
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        target = board[ny][nx]
                        if target and target.owner == opponent and target.side == "stone":
                            # Check if push destination is valid
                            px, py = nx + dx, ny + dy
                            if (in_bounds(px, py, rows, cols) and 
                                board[py][px] is None and
                                not is_opponent_score_cell(px, py, target.owner, rows, cols, score_cols)):
                                my_stone_pushes += 1
            
            # Count RIVER PUSHES (much more powerful!)
            if piece.owner == player and piece.side == "river":
                river_orientation = piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                
                # River can push stones along its flow direction
                if river_orientation == "horizontal":
                    push_directions = [(1, 0), (-1, 0)]  # Left/right
                else:  # vertical
                    push_directions = [(0, 1), (0, -1)]  # Up/down
                
                for dx, dy in push_directions:
                    # Check adjacent position
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        target = board[ny][nx]
                        if target and target.owner == opponent and target.side == "stone":
                            # River can push! Calculate how far
                            push_distance = 0
                            px, py = nx, ny
                            
                            # Keep pushing along river direction until blocked
                            while True:
                                px, py = px + dx, py + dy
                                if not in_bounds(px, py, rows, cols):
                                    break
                                if board[py][px] is not None:
                                    break
                                if is_opponent_score_cell(px, py, target.owner, rows, cols, score_cols):
                                    break
                                push_distance += 1
                                
                                # Limit to reasonable distance (prevent infinite)
                                if push_distance >= 10:
                                    break
                            
                            # Only count if we can actually push (at least 1 space)
                            if push_distance > 0:
                                my_river_pushes += 1
                                my_river_push_distance += push_distance
            
            # Count my vulnerabilities (stones that can be pushed by opponent)
            if piece.owner == player and piece.side == "stone":
                # Vulnerable to opponent stones
                for dx, dy in directions:
                    nx, ny = x - dx, y - dy  # Reverse direction
                    if in_bounds(nx, ny, rows, cols):
                        pusher = board[ny][nx]
                        if pusher and pusher.owner == opponent and pusher.side == "stone":
                            # Check if I can be pushed
                            px, py = x + dx, y + dy
                            if (in_bounds(px, py, rows, cols) and
                                board[py][px] is None and
                                not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                                my_vulnerabilities += 1
                
                # Vulnerable to opponent rivers (more dangerous!)
                for dx, dy in directions:
                    nx, ny = x - dx, y - dy  # Reverse direction
                    if in_bounds(nx, ny, rows, cols):
                        pusher = board[ny][nx]
                        if pusher and pusher.owner == opponent and pusher.side == "river":
                            river_orientation = pusher.orientation if hasattr(pusher, 'orientation') else "horizontal"
                            # Check if river can push in this direction
                            can_push = False
                            if river_orientation == "horizontal" and dx != 0:
                                can_push = True
                            elif river_orientation == "vertical" and dy != 0:
                                can_push = True
                            
                            if can_push:
                                # Check if destination is valid
                                px, py = x + dx, y + dy
                                if (in_bounds(px, py, rows, cols) and
                                    board[py][px] is None and
                                    not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                                    my_vulnerabilities += 2  # River pushes are 2x more dangerous
    
    # Normalize advantage
    # River pushes are worth 5x stone pushes (due to multi-space disruption)
    my_total_push_power = my_stone_pushes + (my_river_pushes * 5) + my_river_push_distance
    max_pushes = 12 * 4 * 6  # Max pieces * directions * average power
    
    advantage = (my_total_push_power - my_vulnerabilities * 3) / max_pushes
    
    return {
        "my_stone_pushes": my_stone_pushes,
        "my_river_pushes": my_river_pushes,
        "my_river_push_distance": my_river_push_distance,
        "my_vulnerabilities": my_vulnerabilities,
        "advantage": advantage
    }


def evaluate_lane_control(board: List[List[Any]], player: str, rows: int, cols: int,
                          score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 2: Evaluate control of scoring lanes (the 4 columns).
    
    Only 4 columns matter for scoring. Control them to win.
    
    Returns:
        {
            "my_lane_pieces": int - My pieces in scoring lanes,
            "opp_lane_pieces": int,
            "advantage": float - Normalized
        }
    """
    opponent = get_opponent(player)
    
    my_lane_pieces = 0
    opp_lane_pieces = 0
    
    for y in range(rows):
        for x in score_cols:
            piece = board[y][x]
            if piece:
                if piece.owner == player:
                    my_lane_pieces += 1
                else:
                    opp_lane_pieces += 1
    
    # Normalize by max possible (4 lanes * rows)
    max_in_lanes = len(score_cols) * rows
    advantage = (my_lane_pieces - opp_lane_pieces) / max_in_lanes
    
    return {
        "my_lane_pieces": my_lane_pieces,
        "opp_lane_pieces": opp_lane_pieces,
        "advantage": advantage
    }


# ==================== TIER 3 EVALUATIONS (POSITIONAL) ====================
# These evaluations measure positional advantages

def evaluate_defensive_barriers(board: List[List[Any]], player: str, rows: int, cols: int,
                                score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 3: Evaluate defensive river barriers.
    
    Enhanced to check if rivers actually BLOCK opponent paths to goal.
    Rivers must be positioned to intercept opponent stones' advancement.
    
    Returns:
        {
            "my_blocking_rivers": int - Rivers that block opponent paths,
            "opp_blocking_rivers": int - Opponent rivers blocking my paths,
            "my_barrier_quality": float - How effective the barriers are,
            "advantage": float
        }
    """
    opponent = get_opponent(player)
    
    my_goal_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    opp_goal_row = bottom_score_row(rows) if player == "circle" else top_score_row()
    
    my_blocking_rivers = 0
    opp_blocking_rivers = 0
    my_barrier_quality = 0.0
    opp_barrier_quality = 0.0
    
    # Find opponent stones and check if my rivers block their paths
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue
            
            # Check if my rivers block opponent stones
            if piece.side == "river" and piece.owner == player:
                river_orientation = piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                
                # Check if this river blocks any opponent stones
                blocks_opponent = False
                block_quality = 0
                
                # Look for opponent stones that this river could block
                for oy in range(rows):
                    for ox in range(cols):
                        opp_piece = board[oy][ox]
                        if opp_piece and opp_piece.owner == opponent and opp_piece.side == "stone":
                            # Check if river is between opponent stone and their goal
                            stone_to_goal_distance = abs(oy - opp_goal_row)
                            river_to_goal_distance = abs(y - opp_goal_row)
                            
                            # River must be closer to goal than the stone (blocking path)
                            if river_to_goal_distance < stone_to_goal_distance:
                                # Check if river is in the stone's column or nearby
                                col_distance = abs(x - ox)
                                
                                # Horizontal river blocks vertical movement in nearby columns
                                if river_orientation == "horizontal" and col_distance <= 1:
                                    blocks_opponent = True
                                    # Quality based on how close to opponent's path
                                    block_quality += 1.0 / max(1, col_distance + 1)
                                
                                # Vertical river blocks horizontal movement in same row
                                elif river_orientation == "vertical" and abs(y - oy) <= 1:
                                    blocks_opponent = True
                                    block_quality += 1.0 / max(1, abs(y - oy) + 1)
                
                if blocks_opponent:
                    my_blocking_rivers += 1
                    my_barrier_quality += min(block_quality, 3.0)  # Cap quality per river
            
            # Check if opponent rivers block my stones
            elif piece.side == "river" and piece.owner == opponent:
                river_orientation = piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                
                blocks_me = False
                block_quality = 0
                
                # Look for my stones that this river could block
                for my_y in range(rows):
                    for my_x in range(cols):
                        my_piece = board[my_y][my_x]
                        if my_piece and my_piece.owner == player and my_piece.side == "stone":
                            stone_to_goal_distance = abs(my_y - my_goal_row)
                            river_to_goal_distance = abs(y - my_goal_row)
                            
                            if river_to_goal_distance < stone_to_goal_distance:
                                col_distance = abs(x - my_x)
                                
                                if river_orientation == "horizontal" and col_distance <= 1:
                                    blocks_me = True
                                    block_quality += 1.0 / max(1, col_distance + 1)
                                
                                elif river_orientation == "vertical" and abs(y - my_y) <= 1:
                                    blocks_me = True
                                    block_quality += 1.0 / max(1, abs(y - my_y) + 1)
                
                if blocks_me:
                    opp_blocking_rivers += 1
                    opp_barrier_quality += min(block_quality, 3.0)
    
    # Normalize quality (max ~36 if all 12 pieces block perfectly)
    my_barrier_quality = my_barrier_quality / 36.0
    opp_barrier_quality = opp_barrier_quality / 36.0
    
    # Advantage combines count and quality
    advantage = (my_barrier_quality - opp_barrier_quality)
    
    return {
        "my_blocking_rivers": my_blocking_rivers,
        "opp_blocking_rivers": opp_blocking_rivers,
        "my_barrier_quality": my_barrier_quality,
        "advantage": advantage
    }


# ==================== TIER 4 EVALUATIONS (STRATEGIC) ====================
# These evaluations measure long-term strategic factors

def evaluate_piece_balance(board: List[List[Any]], player: str, rows: int, cols: int,
                           score_cols: List[int], constants: GameConstants) -> Dict[str, Any]:
    """
    TIER 4: Evaluate stone vs river balance.
    
    Need stones to score, rivers to move. Balance matters.
    
    Returns:
        {
            "my_stones": int,
            "my_rivers": int,
            "my_ratio": float - stones / total,
            "opp_ratio": float,
            "advantage": float
        }
    """
    opponent = get_opponent(player)
    
    my_stones = 0
    my_rivers = 0
    opp_stones = 0
    opp_rivers = 0
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece:
                if piece.owner == player:
                    if piece.side == "stone":
                        my_stones += 1
                    else:
                        my_rivers += 1
                else:
                    if piece.side == "stone":
                        opp_stones += 1
                    else:
                        opp_rivers += 1
    
    my_total = my_stones + my_rivers
    opp_total = opp_stones + opp_rivers
    
    my_ratio = my_stones / max(1, my_total)
    opp_ratio = opp_stones / max(1, opp_total)
    
    # Ideal ratio is ~0.7 (more stones than rivers)
    ideal = 0.7
    my_balance = 1.0 - abs(my_ratio - ideal)
    opp_balance = 1.0 - abs(opp_ratio - ideal)
    
    advantage = my_balance - opp_balance
    
    return {
        "my_stones": my_stones,
        "my_rivers": my_rivers,
        "my_ratio": my_ratio,
        "opp_ratio": opp_ratio,
        "advantage": advantage
    }


def evaluate_position(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    """
    COMPREHENSIVE POSITION EVALUATION using refined tier system.
    
    This is called hundreds of times during minimax search.
    Uses game-derived weights - NO HARDCODED ARBITRARY NUMBERS!
    
    REFINED Tier Weights (after removing Mobility & Tempo):
    - Tier 1 (Game-Decisive): 75% - Material + Distance + Can Win
    - Tier 2 (Tactical): 22% - River Networks + River Pushes + Lane Control
    - Tier 3 (Positional): 3% - Defensive Barriers only
    - Tier 4 (Strategic): Minimal - Piece Balance for fine-tuning
    
    Returns:
        Float score where positive = good for player, negative = bad
    """
    # Initialize constants
    constants = GameConstants(rows, cols)
    
    # ===== TIER 1: GAME-DECISIVE (75% weight = ~1500 points) =====
    material = evaluate_material_normalized(board, player, rows, cols, score_cols, constants)
    distance = evaluate_distance_normalized(board, player, rows, cols, score_cols, constants)
    
    # Material is MOST important (50% of total = 1000 points)
    tier1_score = material["advantage"] * 1000.0
    
    # Distance is secondary (25% of total = 500 points)
    tier1_score += distance["advantage"] * 500.0
    
    # ===== TIER 2: TACTICAL (22% weight = ~440 points) =====
    rivers = evaluate_river_networks(board, player, rows, cols, score_cols, constants)
    pushes = evaluate_push_opportunities(board, player, rows, cols, score_cols, constants)
    lanes = evaluate_lane_control(board, player, rows, cols, score_cols, constants)
    
    tier2_score = 0.0
    tier2_score += rivers["advantage"] * 180.0   # River highways (9% of total)
    tier2_score += pushes["advantage"] * 180.0   # Push power (9% of total) - enhanced with river pushes!
    tier2_score += lanes["advantage"] * 80.0     # Lane control (4% of total)
    
    # ===== TIER 3: POSITIONAL (3% weight = ~60 points) =====
    barriers = evaluate_defensive_barriers(board, player, rows, cols, score_cols, constants)
    
    tier3_score = barriers["advantage"] * 60.0   # Defensive barriers only
    
    # ===== TIER 4: STRATEGIC (minimal = ~10 points) =====
    balance = evaluate_piece_balance(board, player, rows, cols, score_cols, constants)
    tier4_score = balance["advantage"] * 10.0
    
    # ===== COMBINE ALL TIERS =====
    # Total possible range: ~2010 points
    # Tier 1: 1500 (75%)
    # Tier 2: 440 (22%)
    # Tier 3: 60 (3%)
    # Tier 4: 10 (0.5%)
    total_score = tier1_score + tier2_score + tier3_score + tier4_score
    
    return total_score


def assess_opponent_threats(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> Dict[str, Any]:
    """
    LEGACY FUNCTION: Wrapper for backward compatibility.
    Use evaluate_can_win_this_turn() and other Tier 1 evaluations instead.
    
    Analyze opponent's threatening positions.
    
    Returns a dictionary with threat analysis:
    - "can_win_next_turn": bool - Opponent has 3 stones and can score 4th
    - "scoring_stones": int - Number of opponent stones in their scoring area
    - "closest_threat_distance": int - Distance of closest opponent stone to their goal
    - "threats_near_goal": List - Opponent pieces within 2 moves of scoring
    """
    constants = GameConstants(rows, cols)
    opponent = get_opponent(player)
    
    # Use new evaluation functions
    opp_win = evaluate_can_win_this_turn(board, opponent, rows, cols, score_cols, constants)
    material = evaluate_material_normalized(board, opponent, rows, cols, score_cols, constants)
    distance = evaluate_distance_normalized(board, opponent, rows, cols, score_cols, constants)
    
    # Find threatening pieces
    threats_near_goal = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == opponent and piece.side == "stone":
                dist = calculate_distance_to_scoring(x, y, opponent, rows, cols, score_cols)
                if dist <= constants.THREAT_DISTANCE_NEAR:
                    threats_near_goal.append({
                        "position": (x, y),
                        "distance": dist
                    })
    
    return {
        "can_win_next_turn": opp_win["can_win"],
        "scoring_stones": material["opp_stones"],
        "closest_threat_distance": distance["opp_closest"],
        "threats_near_goal": threats_near_goal
    }


def can_we_win_this_turn(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """
    LEGACY FUNCTION: Wrapper for backward compatibility.
    Use evaluate_can_win_this_turn() instead.
    
    Check if we have 3 stones in scoring area and can potentially score the 4th this turn.
    """
    constants = GameConstants(rows, cols)
    result = evaluate_can_win_this_turn(board, player, rows, cols, score_cols, constants)
    return result["can_win"]


def get_comprehensive_evaluation(board: List[List[Any]], player: str, rows: int, cols: int, 
                                 score_cols: List[int]) -> Dict[str, Any]:
    """
    Get ALL evaluations at once (useful for debugging).
    
    Returns a complete breakdown of the position across all refined tiers.
    """
    constants = GameConstants(rows, cols)
    
    return {
        "tier1_material": evaluate_material_normalized(board, player, rows, cols, score_cols, constants),
        "tier1_distance": evaluate_distance_normalized(board, player, rows, cols, score_cols, constants),
        "tier1_can_win": evaluate_can_win_this_turn(board, player, rows, cols, score_cols, constants),
        "tier2_rivers": evaluate_river_networks(board, player, rows, cols, score_cols, constants),
        "tier2_pushes": evaluate_push_opportunities(board, player, rows, cols, score_cols, constants),
        "tier2_lanes": evaluate_lane_control(board, player, rows, cols, score_cols, constants),
        "tier3_barriers": evaluate_defensive_barriers(board, player, rows, cols, score_cols, constants),
        "tier4_balance": evaluate_piece_balance(board, player, rows, cols, score_cols, constants),
        "overall_score": evaluate_position(board, player, rows, cols, score_cols)
    }



    """
    Analyze opponent's threatening positions.
    
    Returns a dictionary with threat analysis:
    - "can_win_next_turn": bool - Opponent has 3 stones and can score 4th
    - "scoring_stones": int - Number of opponent stones in their scoring area
    - "closest_threat_distance": int - Distance of closest opponent stone to their goal
    - "threats_near_goal": List - Opponent pieces within 2 moves of scoring
    """
    opponent = get_opponent(player)
    
    # Count opponent's scoring stones
    opp_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    # Check if opponent can win next turn (has 3 stones already)
    can_win_next = (opp_scoring_stones == 3)
    
    # Find opponent's closest stone to their goal
    closest_distance = float('inf')
    threats_near_goal = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == opponent and piece.side == "stone":
                distance = calculate_distance_to_scoring(x, y, opponent, rows, cols, score_cols)
                
                if distance < closest_distance:
                    closest_distance = distance
                
                # Pieces within 2 moves are immediate threats
                if distance <= 2:
                    threats_near_goal.append({
                        "position": (x, y),
                        "distance": distance
                    })
    
    return {
        "can_win_next_turn": can_win_next,
        "scoring_stones": opp_scoring_stones,
        "closest_threat_distance": closest_distance if closest_distance != float('inf') else 0,
        "threats_near_goal": threats_near_goal
    }


def can_we_win_this_turn(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """
    Check if we have 3 stones in scoring area and can potentially score the 4th this turn.
    """
    my_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    
    if my_scoring_stones != 3:
        return False
    
    # Check if we have any stone within 1 move of scoring area
    if player == "circle":
        goal_row = top_score_row()
    else:
        goal_row = bottom_score_row(rows)
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Check if this stone is adjacent to an empty scoring cell
                for sx in score_cols:
                    if abs(x - sx) + abs(y - goal_row) == 1:
                        # Check if scoring cell is empty
                        if board[goal_row][sx] is None:
                            return True
    
    return False


# ==================== LAYER 5: STRATEGIC MODE DETERMINATION ====================
# This layer determines overall strategy based on board evaluation

def determine_strategic_mode(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> str:
    """
    Determine the strategic mode using MULTI-DIMENSIONAL evaluation.
    NO HARDCODED ARBITRARY NUMBERS - all thresholds derived from game rules!
    
    Strategic Modes:
    - "CAN_WIN": We can win this turn (always prioritize!)
    - "DESPERATE_DEFENSE": Opponent can win next turn - MUST defend
    - "DEFENSIVE": We're losing badly - focus on defense
    - "PROTECT_LEAD": We're winning strongly - protect position
    - "AGGRESSIVE": We're slightly ahead - press advantage
    - "BALANCED": Even game - mix of advance and attack
    """
    # Initialize constants
    constants = GameConstants(rows, cols)
    
    # Check if we can win this turn (highest priority)
    my_win = evaluate_can_win_this_turn(board, player, rows, cols, score_cols, constants)
    if my_win["can_win"]:
        return "CAN_WIN"
    
    # Check if opponent can win next turn
    opponent = get_opponent(player)
    opp_win = evaluate_can_win_this_turn(board, opponent, rows, cols, score_cols, constants)
    if opp_win["can_win"]:
        return "DESPERATE_DEFENSE"
    
    # Get comprehensive evaluations (material + distance only, no tempo)
    material = evaluate_material_normalized(board, player, rows, cols, score_cols, constants)
    distance = evaluate_distance_normalized(board, player, rows, cols, score_cols, constants)
    
    # Decision based on material advantage (in terms of stones, not arbitrary numbers)
    stone_diff = material["my_stones"] - material["opp_stones"]
    
    # Big advantage: >= 50% of winning condition (2+ stones)
    if stone_diff >= constants.STRATEGIC_ADVANTAGE_BIG:
        return "PROTECT_LEAD"
    
    # Big disadvantage: <= -50% of winning condition
    elif stone_diff <= -constants.STRATEGIC_ADVANTAGE_BIG:
        return "DEFENSIVE"
    
    # Small advantage: use combined material + distance to decide
    elif stone_diff >= constants.STRATEGIC_ADVANTAGE_SMALL:
        # We're ahead - check distance advantage too
        combined_advantage = material["advantage"] + distance["advantage"]
        if combined_advantage > 0.3:  # Strong overall position
            return "AGGRESSIVE"
        else:
            return "BALANCED"
    
    # Small disadvantage
    elif stone_diff <= -constants.STRATEGIC_ADVANTAGE_SMALL:
        return "DEFENSIVE"
    
    # Material tied - use distance to decide
    else:
        if distance["advantage"] > 0.2:  # We're closer to goal
            return "AGGRESSIVE"
        elif distance["advantage"] < -0.2:  # Opponent is closer
            return "DEFENSIVE"
        else:
            return "BALANCED"


# ==================== COMPREHENSIVE MOVE ANALYSIS ====================
# Single-pass analysis that generates moves and evaluates them together

def analyze_position_with_moves(board: List[List[Any]], player: str, rows: int, cols: int,
                                 score_cols: List[int]) -> Dict[str, Any]:
    """
    COMPREHENSIVE ANALYSIS: Scan board ONCE, generate all moves with evaluations.
    
    This replaces duplicate work in evaluation + move generation.
    
    Returns:
        {
            "winning_moves": List[Dict] - Moves that win the game,
            "scoring_moves": List[Dict] - Moves that score stones,
            "advancing_moves": List[Dict] - Moves toward goal,
            "push_moves": List[Dict] - Offensive pushes,
            "defensive_moves": List[Dict] - Counter opponent threats,
            "setup_moves": List[Dict] - Flips and rotates,
            "opponent_threats": Dict - Opponent's dangerous positions,
            "material_eval": Dict - Material evaluation,
            "distance_eval": Dict - Distance evaluation,
            "overall_score": float - Position score
        }
    """
    constants = GameConstants(rows, cols)
    opponent = get_opponent(player)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # Initialize move collections
    winning_moves = []
    scoring_moves = []
    advancing_moves = []
    push_moves = []
    defensive_moves = []
    setup_moves = []
    
    # Track material
    my_stones_in_goal = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opp_stones_in_goal = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    my_goal_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    opp_goal_row = bottom_score_row(rows) if player == "circle" else top_score_row()
    
    # Opponent threat analysis
    opponent_threatening_stones = []
    opponent_rivers = []
    opponent_push_threats = []
    
    # Single board scan
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece:
                continue
            
            # === ANALYZE MY PIECES ===
            if piece.owner == player:
                if piece.side == "stone":
                    # Check for winning/scoring moves
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if not in_bounds(nx, ny, rows, cols):
                            continue
                        if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                            continue
                        
                        target = board[ny][nx]
                        
                        # Simple move
                        if target is None:
                            move = {"action": "move", "from": [x, y], "to": [nx, ny]}
                            
                            # Is it scoring?
                            if ny == my_goal_row and nx in score_cols:
                                if my_stones_in_goal == 3:
                                    winning_moves.append(move)
                                else:
                                    scoring_moves.append(move)
                            # Is it advancing?
                            elif (player == "circle" and ny < y) or (player == "square" and ny > y):
                                advancing_moves.append(move)
                        
                        # Push move
                        elif target.owner == opponent:
                            px, py = nx + dx, ny + dy
                            if (in_bounds(px, py, rows, cols) and 
                                board[py][px] is None and
                                not is_opponent_score_cell(px, py, opponent, rows, cols, score_cols)):
                                push_moves.append({
                                    "action": "push",
                                    "from": [x, y],
                                    "to": [nx, ny],
                                    "pushed_to": [px, py]
                                })
                    
                    # SMART SETUP: Only flip stones that are strategically positioned
                    # 1. Stones near goal (can create river highway)
                    # 2. Stones that could connect to existing rivers
                    # 3. Stones in the middle of the board (flexible positioning)
                    
                    dist_to_goal = abs(y - my_goal_row)
                    is_near_goal = dist_to_goal <= 3
                    is_in_scoring_col = x in score_cols
                    is_central = rows // 3 <= y <= 2 * rows // 3
                    
                    # Check if adjacent to my rivers (could extend network)
                    adjacent_to_river = False
                    for dx, dy in directions:
                        adj_x, adj_y = x + dx, y + dy
                        if in_bounds(adj_x, adj_y, rows, cols):
                            adj_piece = board[adj_y][adj_x]
                            if adj_piece and adj_piece.owner == player and adj_piece.side == "river":
                                adjacent_to_river = True
                                break
                    
                    # Generate strategic flips
                    if is_near_goal or adjacent_to_river or is_central:
                        for orientation in ["horizontal", "vertical"]:
                            move = {
                                "action": "flip",
                                "from": [x, y],
                                "orientation": orientation
                            }
                            # Tag strategic value
                            if is_near_goal and is_in_scoring_col:
                                move["strategic_value"] = "goal_highway"
                            elif adjacent_to_river:
                                move["strategic_value"] = "extend_network"
                            else:
                                move["strategic_value"] = "positioning"
                            setup_moves.append(move)
                
                elif piece.side == "river":
                    # SMART ROTATE: Only rotate if it could improve connectivity or blocking
                    current_ori = piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                    
                    # Check if rotating would connect to more pieces
                    h_connections = 0  # horizontal connections
                    v_connections = 0  # vertical connections
                    
                    # Check horizontal neighbors
                    for dx in [-1, 1]:
                        nx = x + dx
                        if in_bounds(nx, y, rows, cols):
                            neighbor = board[y][nx]
                            if neighbor and neighbor.owner == player:
                                h_connections += 1
                    
                    # Check vertical neighbors
                    for dy in [-1, 1]:
                        ny = y + dy
                        if in_bounds(x, ny, rows, cols):
                            neighbor = board[ny][x]
                            if neighbor and neighbor.owner == player:
                                v_connections += 1
                    
                    # Rotate if it improves alignment
                    if current_ori == "horizontal" and v_connections > h_connections:
                        setup_moves.append({
                            "action": "rotate",
                            "from": [x, y],
                            "strategic_value": "align_network"
                        })
                    elif current_ori == "vertical" and h_connections > v_connections:
                        setup_moves.append({
                            "action": "rotate",
                            "from": [x, y],
                            "strategic_value": "align_network"
                        })
                    
                    # Always allow flip back to stone (might need mobility)
                    setup_moves.append({
                        "action": "flip",
                        "from": [x, y],
                        "strategic_value": "revert_to_stone"
                    })
            
            # === ANALYZE OPPONENT PIECES ===
            else:  # opponent's piece
                if piece.side == "stone":
                    # Track threatening stones
                    dist_to_goal = abs(y - opp_goal_row) + min(abs(x - sc) for sc in score_cols)
                    if dist_to_goal <= 3:
                        opponent_threatening_stones.append({
                            "pos": (x, y),
                            "distance": dist_to_goal
                        })
                    
                    # Check if opponent can push my pieces
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if in_bounds(nx, ny, rows, cols):
                            target = board[ny][nx]
                            if target and target.owner == player and target.side == "stone":
                                px, py = nx + dx, ny + dy
                                if (in_bounds(px, py, rows, cols) and board[py][px] is None):
                                    opponent_push_threats.append({
                                        "my_piece": (nx, ny),
                                        "pusher": (x, y)
                                    })
                
                elif piece.side == "river":
                    opponent_rivers.append({
                        "pos": (x, y),
                        "orientation": piece.orientation if hasattr(piece, 'orientation') else "horizontal"
                    })
    
    # === GENERATE DEFENSIVE MOVES BASED ON THREATS ===
    # Counter opponent's threatening stones
    for threat in opponent_threatening_stones:
        tx, ty = threat["pos"]
        # Try to push them back
        for dx, dy in directions:
            px, py = tx - dx, ty - dy
            if in_bounds(px, py, rows, cols):
                pusher = board[py][px]
                if pusher and pusher.owner == player and pusher.side == "stone":
                    # Can we push this threat?
                    pushed_to_x, pushed_to_y = tx + dx, ty + dy
                    if (in_bounds(pushed_to_x, pushed_to_y, rows, cols) and
                        board[pushed_to_y][pushed_to_x] is None):
                        defensive_moves.append({
                            "action": "push",
                            "from": [px, py],
                            "to": [tx, ty],
                            "pushed_to": [pushed_to_x, pushed_to_y],
                            "reason": "counter_threat"
                        })
    
    # Block opponent rivers with our pieces
    for opp_river in opponent_rivers:
        rx, ry = opp_river["pos"]
        opp_ori = opp_river["orientation"]
        
        # Try to place perpendicular river to block
        for dx, dy in directions:
            bx, by = rx + dx, ry + dy
            if in_bounds(bx, by, rows, cols):
                blocker = board[by][bx]
                if blocker and blocker.owner == player and blocker.side == "stone":
                    # Flip to perpendicular river to block
                    block_ori = "vertical" if opp_ori == "horizontal" else "horizontal"
                    defensive_moves.append({
                        "action": "flip",
                        "from": [bx, by],
                        "orientation": block_ori,
                        "reason": "block_river"
                    })
    
    # Move vulnerable pieces away from push threats
    for threat in opponent_push_threats:
        vx, vy = threat["my_piece"]
        # Find escape moves
        for dx, dy in directions:
            ex, ey = vx + dx, vy + dy
            if in_bounds(ex, ey, rows, cols) and board[ey][ex] is None:
                if not is_opponent_score_cell(ex, ey, player, rows, cols, score_cols):
                    defensive_moves.append({
                        "action": "move",
                        "from": [vx, vy],
                        "to": [ex, ey],
                        "reason": "escape_push"
                    })
    
    # Compute evaluations
    material_eval = evaluate_material_normalized(board, player, rows, cols, score_cols, constants)
    distance_eval = evaluate_distance_normalized(board, player, rows, cols, score_cols, constants)
    overall_score = evaluate_position(board, player, rows, cols, score_cols)
    
    return {
        "winning_moves": winning_moves,
        "scoring_moves": scoring_moves,
        "advancing_moves": advancing_moves,
        "push_moves": push_moves,
        "defensive_moves": defensive_moves,
        "setup_moves": setup_moves,
        "opponent_threats": {
            "threatening_stones": opponent_threatening_stones,
            "rivers": opponent_rivers,
            "push_threats": opponent_push_threats
        },
        "material_eval": material_eval,
        "distance_eval": distance_eval,
        "overall_score": overall_score
    }


# ==================== LAYER 6: TARGETED MOVE GENERATION ====================
# This layer generates ONLY moves relevant to the current strategy

def generate_winning_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate ONLY moves that win the game immediately.
    (Moves that place our 4th stone in scoring area)
    """
    all_moves = generate_all_moves(board, player, rows, cols, score_cols)
    winning_moves = []
    
    for move in all_moves:
        category = categorize_move(move, board, player, rows, cols, score_cols)
        if category == "WINNING":
            winning_moves.append(move)
    
    return winning_moves


def generate_defensive_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate moves focused on defense:
    - Push opponent's threatening stones backward
    - Flip opponent stones to rivers
    - Block opponent's path to scoring area
    """
    opponent = get_opponent(player)
    threats = assess_opponent_threats(board, player, rows, cols, score_cols)
    defensive_moves = []
    
    # Generate all moves
    all_moves = generate_all_moves(board, player, rows, cols, score_cols)
    
    # Prioritize moves that attack opponent's threatening pieces
    for move in all_moves:
        action = move.get("action")
        
        # Pushes are defensive if they push opponent back
        if action == "push":
            to_pos = move.get("to")
            if to_pos:
                tx, ty = to_pos
                target_piece = board[ty][tx]
                if target_piece and target_piece.owner == opponent:
                    # Check if this piece is a threat
                    for threat in threats["threats_near_goal"]:
                        if threat["position"] == (tx, ty):
                            defensive_moves.append(move)
                            break
                    else:
                        # Still add push moves as defensive
                        defensive_moves.append(move)
        
        # Also include scoring moves (best defense is good offense)
        category = categorize_move(move, board, player, rows, cols, score_cols)
        if category in ["WINNING", "SCORING"]:
            defensive_moves.append(move)
    
    # If no specific defensive moves found, return some attacking moves
    if not defensive_moves:
        for move in all_moves:
            category = categorize_move(move, board, player, rows, cols, score_cols)
            if category in ["ATTACKING", "ADVANCING"]:
                defensive_moves.append(move)
    
    return defensive_moves if defensive_moves else all_moves


def generate_offensive_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate moves focused on offense:
    - Advance our stones toward scoring area
    - Score stones in scoring area
    - Flip rivers to stones for advancement
    - Attack opponent to slow them down
    """
    all_moves = generate_all_moves(board, player, rows, cols, score_cols)
    offensive_moves = []
    
    for move in all_moves:
        category = categorize_move(move, board, player, rows, cols, score_cols)
        
        # Prioritize offensive categories
        if category in ["WINNING", "SCORING", "ADVANCING", "ATTACKING", "SETUP"]:
            offensive_moves.append(move)
    
    return offensive_moves if offensive_moves else all_moves


def generate_balanced_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate a balanced mix of offensive and defensive moves.
    """
    return generate_all_moves(board, player, rows, cols, score_cols)


# ==================== LAYER 7: MOVE SELECTION ====================
# This layer implements the final decision-making strategy

def select_best_move(moves: List[Dict[str, Any]], board: List[List[Any]], 
                     player: str, rows: int, cols: int, score_cols: List[int]) -> Dict[str, Any]:
    """
    LAYER 5: Select the best move from the available moves.
    
    Strategy:
    1. Score all moves using Layer 3
    2. Pick the highest-scoring move
    3. If multiple moves have the same top score, pick randomly among them
    """
    if not moves:
        return None
    
    # Score all moves
    scored_moves = rank_moves_by_score(moves, board, player, rows, cols, score_cols)
    
    # Get the best score
    best_score = scored_moves[0][1]
    
    # Find all moves with the best score (in case of ties)
    best_moves = [move for move, score in scored_moves if score == best_score]
    
    # Randomly select among equally good moves
    return random.choice(best_moves)


# ==================== LAYER 8: ZOBRIST HASHING ====================
# Fast board state hashing for transposition tables

class ZobristHash:
    """
    Zobrist hashing for fast board state comparison.
    
    Each board position gets a unique hash by XORing random numbers
    for each piece on each position. This allows O(1) position lookups
    in the transposition table.
    """
    
    def __init__(self, rows: int, cols: int):
        """Initialize Zobrist random numbers for all possible piece placements."""
        self.rows = rows
        self.cols = cols
        
        # Random number for each: (x, y, player, side, orientation)
        # We need random numbers for:
        # - Each position (x, y)
        # - Each player (circle, square)
        # - Each side (stone, river)
        # - Each orientation (horizontal, vertical) for rivers
        
        random.seed(42)  # Fixed seed for reproducibility
        
        self.zobrist_table = {}
        
        # Generate random numbers for all combinations
        for y in range(rows):
            for x in range(cols):
                for player in ["circle", "square"]:
                    # Stone pieces
                    key = (x, y, player, "stone", None)
                    self.zobrist_table[key] = random.randint(0, 2**64 - 1)
                    
                    # River pieces with orientations
                    for orientation in ["horizontal", "vertical"]:
                        key = (x, y, player, "river", orientation)
                        self.zobrist_table[key] = random.randint(0, 2**64 - 1)
    
    def hash_board(self, board: List[List[Any]]) -> int:
        """
        Compute Zobrist hash for current board state.
        
        Returns:
            64-bit integer hash of the board position
        """
        hash_value = 0
        
        for y in range(self.rows):
            for x in range(self.cols):
                piece = board[y][x]
                if piece:
                    key = (x, y, piece.owner, piece.side, piece.orientation)
                    hash_value ^= self.zobrist_table.get(key, 0)
        
        return hash_value


# ==================== LAYER 9: TRANSPOSITION TABLE ====================
# Cache board evaluations to avoid recomputing same positions

class TranspositionTable:
    """
    Stores previously evaluated board positions.
    
    Key benefits:
    - Avoid re-evaluating the same position
    - Store best moves found for positions
    - Prune search tree more effectively
    """
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize transposition table.
        
        Args:
            max_size: Maximum number of entries (prevents memory overflow)
        """
        self.table = {}
        self.max_size = max_size
    
    def store(self, board_hash: int, depth: int, score: float, best_move: Optional[Dict[str, Any]], flag: str):
        """
        Store a board evaluation.
        
        Args:
            board_hash: Zobrist hash of the board
            depth: Search depth this was evaluated at
            score: Evaluation score
            best_move: Best move found for this position
            flag: Type of bound - "EXACT", "LOWER", "UPPER"
        """
        if len(self.table) >= self.max_size:
            # Simple eviction: remove oldest entry
            self.table.pop(next(iter(self.table)))
        
        self.table[board_hash] = {
            "depth": depth,
            "score": score,
            "best_move": best_move,
            "flag": flag
        }
    
    def lookup(self, board_hash: int, depth: int, alpha: float, beta: float) -> Optional[Tuple[float, Optional[Dict[str, Any]]]]:
        """
        Look up a board position.
        
        Args:
            board_hash: Zobrist hash to look up
            depth: Current search depth
            alpha, beta: Alpha-beta bounds
        
        Returns:
            (score, best_move) if usable, None otherwise
        """
        entry = self.table.get(board_hash)
        
        if entry is None:
            return None
        
        # Only use if depth is sufficient
        if entry["depth"] < depth:
            return None
        
        score = entry["score"]
        flag = entry["flag"]
        
        # Check if we can use this score based on bounds
        if flag == "EXACT":
            return (score, entry["best_move"])
        elif flag == "LOWER" and score >= beta:
            return (score, entry["best_move"])
        elif flag == "UPPER" and score <= alpha:
            return (score, entry["best_move"])
        
        return None
    
    def get_best_move(self, board_hash: int) -> Optional[Dict[str, Any]]:
        """Get the best move for a position (if cached)."""
        entry = self.table.get(board_hash)
        return entry["best_move"] if entry else None
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()


# ==================== LAYER 10: MOVE ORDERING ====================
# Order moves to improve alpha-beta pruning efficiency

def order_moves_for_search(moves: List[Dict[str, Any]], board: List[List[Any]], 
                           player: str, rows: int, cols: int, score_cols: List[int],
                           transposition_table: Optional[TranspositionTable] = None,
                           zobrist: Optional[ZobristHash] = None) -> List[Dict[str, Any]]:
    """
    Order moves to search best ones first (improves alpha-beta pruning).
    
    Move ordering heuristics:
    1. Transposition table best move (if available)
    2. Winning moves
    3. Scoring moves
    4. Capturing/pushing moves
    5. Advancing moves
    6. Other moves
    
    Args:
        moves: List of moves to order
        board: Current board state
        player: Current player
        transposition_table: Optional TT for best move hint
        zobrist: Optional Zobrist hasher
    
    Returns:
        Ordered list of moves (best first)
    """
    if not moves:
        return moves
    
    # Try to get best move from transposition table
    tt_best_move = None
    if transposition_table and zobrist:
        board_hash = zobrist.hash_board(board)
        tt_best_move = transposition_table.get_best_move(board_hash)
    
    # Score each move for ordering
    move_scores = []
    
    for move in moves:
        score = 0
        
        # Highest priority: TT best move
        if tt_best_move and move == tt_best_move:
            score = 1000000
        else:
            # Use move evaluation scores
            category = categorize_move(move, board, player, rows, cols, score_cols)
            
            if category == "WINNING":
                score = 10000
            elif category == "SCORING":
                score = 1000
            elif category == "ATTACKING":
                score = 500
            elif category == "ADVANCING":
                score = 100
            elif category == "SETUP":
                score = 50
            else:
                score = 10
        
        move_scores.append((move, score))
    
    # Sort by score (highest first)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [move for move, score in move_scores]


# ==================== LAYER 11: MINIMAX WITH ALPHA-BETA PRUNING ====================
# Game tree search with alpha-beta pruning

def minimax_alpha_beta(board: List[List[Any]], depth: int, alpha: float, beta: float,
                       maximizing_player: bool, player: str, rows: int, cols: int, 
                       score_cols: List[int], zobrist: ZobristHash, 
                       transposition_table: TranspositionTable,
                       start_time: float, time_limit: float) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Minimax search with alpha-beta pruning.
    
    This is the CORE search algorithm that looks ahead to find the best move.
    
    Algorithm:
    1. If depth = 0 or game over: return evaluation
    2. Generate all legal moves
    3. Order moves (best first for pruning)
    4. For each move:
       - Apply move to board
       - Recursively search opponent's response
       - Undo move
       - Update alpha/beta bounds
       - Prune if possible
    5. Return best score and move
    
    Args:
        board: Current board state
        depth: Remaining search depth
        alpha: Best score for maximizer
        beta: Best score for minimizer
        maximizing_player: True if current player is maximizing
        player: Current player making the move
        zobrist: Zobrist hasher
        transposition_table: Cache of evaluated positions
        start_time: When search started (for time management)
        time_limit: Maximum time allowed (seconds)
    
    Returns:
        (best_score, best_move) tuple
    """
    # Check time limit
    if time.time() - start_time > time_limit:
        # Time's up! Return quick evaluation
        return evaluate_position(board, player, rows, cols, score_cols), None
    
    # Check transposition table
    board_hash = zobrist.hash_board(board)
    tt_result = transposition_table.lookup(board_hash, depth, alpha, beta)
    if tt_result:
        return tt_result
    
    # Check terminal conditions
    # Check if game is won
    for p in ["circle", "square"]:
        score_count = count_stones_in_scoring_area(board, p, rows, cols, score_cols)
        if score_count >= 4:
            # Game over - return huge score
            if p == player:
                return 100000.0, None  # We won!
            else:
                return -100000.0, None  # We lost!
    
    # Base case: depth = 0, return evaluation
    if depth == 0:
        eval_score = evaluate_position(board, player, rows, cols, score_cols)
        transposition_table.store(board_hash, depth, eval_score, None, "EXACT")
        return eval_score, None
    
    # Generate moves based on strategic mode (targeted generation)
    strategic_mode = determine_strategic_mode(board, player, rows, cols, score_cols)
    
    if strategic_mode == "CAN_WIN":
        moves = generate_winning_moves(board, player, rows, cols, score_cols)
    elif strategic_mode == "DESPERATE_DEFENSE":
        moves = generate_defensive_moves(board, player, rows, cols, score_cols)
    elif strategic_mode == "DEFENSIVE":
        moves = generate_defensive_moves(board, player, rows, cols, score_cols)
    else:
        # For other modes, generate all moves
        moves = generate_all_moves(board, player, rows, cols, score_cols)
    
    if not moves:
        # No moves available
        eval_score = evaluate_position(board, player, rows, cols, score_cols)
        return eval_score, None
    
    # Order moves for better pruning
    moves = order_moves_for_search(moves, board, player, rows, cols, score_cols, 
                                   transposition_table, zobrist)
    
    best_move = None
    opponent_player = get_opponent(player)
    
    if maximizing_player:
        max_eval = float('-inf')
        
        for move in moves:
            # Apply move
            board_copy = copy.deepcopy(board)
            success, _ = apply_move_to_board(board_copy, move, player, rows, cols, score_cols)
            
            if not success:
                continue
            
            # Recursively evaluate opponent's response
            eval_score, _ = minimax_alpha_beta(
                board_copy, depth - 1, alpha, beta, False,
                opponent_player, rows, cols, score_cols,
                zobrist, transposition_table, start_time, time_limit
            )
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break  # Beta cutoff
        
        # Store in transposition table
        flag = "EXACT"
        if max_eval <= alpha:
            flag = "UPPER"
        elif max_eval >= beta:
            flag = "LOWER"
        
        transposition_table.store(board_hash, depth, max_eval, best_move, flag)
        return max_eval, best_move
    
    else:  # Minimizing player
        min_eval = float('inf')
        
        for move in moves:
            # Apply move
            board_copy = copy.deepcopy(board)
            success, _ = apply_move_to_board(board_copy, move, player, rows, cols, score_cols)
            
            if not success:
                continue
            
            # Recursively evaluate opponent's response
            eval_score, _ = minimax_alpha_beta(
                board_copy, depth - 1, alpha, beta, True,
                opponent_player, rows, cols, score_cols,
                zobrist, transposition_table, start_time, time_limit
            )
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break  # Alpha cutoff
        
        # Store in transposition table
        flag = "EXACT"
        if min_eval <= alpha:
            flag = "UPPER"
        elif min_eval >= beta:
            flag = "LOWER"
        
        transposition_table.store(board_hash, depth, min_eval, best_move, flag)
        return min_eval, best_move


def apply_move_to_board(board: List[List[Any]], move: Dict[str, Any], player: str,
                        rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, str]:
    """
    Apply a move to the board (in-place modification).
    
    This is a simplified version that doesn't validate as strictly as gameEngine.
    Used for minimax search speed.
    
    Returns:
        (success: bool, message: str)
    """
    action = move.get("action")
    
    try:
        if action == "move":
            from_pos = move.get("from")
            to_pos = move.get("to")
            if not from_pos or not to_pos:
                return False, "Invalid format"
            
            fx, fy = from_pos
            tx, ty = to_pos
            
            if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols)):
                return False, "Out of bounds"
            
            piece = board[fy][fx]
            if not piece or piece.owner != player:
                return False, "Invalid piece"
            
            board[ty][tx] = piece
            board[fy][fx] = None
            return True, "OK"
        
        elif action == "push":
            from_pos = move.get("from")
            to_pos = move.get("to")
            pushed_to = move.get("pushed_to")
            
            if not (from_pos and to_pos and pushed_to):
                return False, "Invalid format"
            
            fx, fy = from_pos
            tx, ty = to_pos
            px, py = pushed_to
            
            if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols) 
                   and in_bounds(px, py, rows, cols)):
                return False, "Out of bounds"
            
            pusher = board[fy][fx]
            target = board[ty][tx]
            
            if not (pusher and target and pusher.owner == player):
                return False, "Invalid push"
            
            board[py][px] = target
            board[ty][tx] = pusher
            board[fy][fx] = None
            
            if pusher.side == "river":
                board[ty][tx].side = "stone"
                board[ty][tx].orientation = None
            
            return True, "OK"
        
        elif action == "flip":
            from_pos = move.get("from")
            if not from_pos:
                return False, "Invalid format"
            
            fx, fy = from_pos
            if not in_bounds(fx, fy, rows, cols):
                return False, "Out of bounds"
            
            piece = board[fy][fx]
            if not piece or piece.owner != player:
                return False, "Invalid piece"
            
            if piece.side == "stone":
                orientation = move.get("orientation", "horizontal")
                piece.side = "river"
                piece.orientation = orientation
            else:
                piece.side = "stone"
                piece.orientation = None
            
            return True, "OK"
        
        elif action == "rotate":
            from_pos = move.get("from")
            if not from_pos:
                return False, "Invalid format"
            
            fx, fy = from_pos
            if not in_bounds(fx, fy, rows, cols):
                return False, "Out of bounds"
            
            piece = board[fy][fx]
            if not piece or piece.owner != player or piece.side != "river":
                return False, "Invalid rotate"
            
            piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
            return True, "OK"
        
        return False, "Unknown action"
    
    except Exception as e:
        return False, str(e)


def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    """
    Simulate a move on a copy of the board.
    
    Args:
        board: Current board state
        move: Move to simulate
        player: Player making the move
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        (success: bool, new_board_state or error_message)
    """
    # Create a deep copy of the board for simulation
    board_copy = copy.deepcopy(board)
    
    # Apply the move to the copy
    action = move.get("action")
    
    try:
        if action == "move":
            from_pos = move.get("from")
            to_pos = move.get("to")
            if not from_pos or not to_pos:
                return False, "Invalid move format"
            
            fx, fy = from_pos
            tx, ty = to_pos
            
            # Validate bounds
            if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols)):
                return False, "Out of bounds"
            
            # Move piece
            piece = board_copy[fy][fx]
            if piece and piece.owner == player:
                board_copy[ty][tx] = piece
                board_copy[fy][fx] = None
                return True, board_copy
            
            return False, "Invalid piece"
        
        elif action == "push":
            from_pos = move.get("from")
            to_pos = move.get("to")
            pushed_to = move.get("pushed_to")
            
            if not from_pos or not to_pos or not pushed_to:
                return False, "Invalid push format"
            
            fx, fy = from_pos
            tx, ty = to_pos
            px, py = pushed_to
            
            # Validate bounds
            if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols) and in_bounds(px, py, rows, cols)):
                return False, "Out of bounds"
            
            # Execute push
            pusher = board_copy[fy][fx]
            target = board_copy[ty][tx]
            
            if pusher and target and pusher.owner == player:
                board_copy[py][px] = target
                board_copy[ty][tx] = pusher
                board_copy[fy][fx] = None
                
                # River that pushes becomes stone
                if pusher.side == "river":
                    board_copy[ty][tx].side = "stone"
                    board_copy[ty][tx].orientation = None
                
                return True, board_copy
            
            return False, "Invalid push"
        
        elif action == "flip":
            from_pos = move.get("from")
            if not from_pos:
                return False, "Invalid flip format"
            
            fx, fy = from_pos
            if not in_bounds(fx, fy, rows, cols):
                return False, "Out of bounds"
            
            piece = board_copy[fy][fx]
            if piece and piece.owner == player:
                if piece.side == "stone":
                    # Flip to river
                    orientation = move.get("orientation", "horizontal")
                    piece.side = "river"
                    piece.orientation = orientation
                else:
                    # Flip to stone
                    piece.side = "stone"
                    piece.orientation = None
                
                return True, board_copy
            
            return False, "Invalid flip"
        
        elif action == "rotate":
            from_pos = move.get("from")
            if not from_pos:
                return False, "Invalid rotate format"
            
            fx, fy = from_pos
            if not in_bounds(fx, fy, rows, cols):
                return False, "Out of bounds"
            
            piece = board_copy[fy][fx]
            if piece and piece.owner == player and piece.side == "river":
                # Toggle orientation
                piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
                return True, board_copy
            
            return False, "Invalid rotate"
        
        else:
            return False, "Unknown action"
    
    except Exception as e:
        return False, str(e)

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation with Minimax Search
    
    This agent combines:
    1. Strategic board evaluation (heuristic)
    2. Minimax search with alpha-beta pruning (look-ahead)
    3. Zobrist hashing (fast position comparison)
    4. Transposition tables (caching)
    
    The bot looks ahead to predict opponent responses and choose
    the move that leads to the best position after opponent plays.
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        
        # Initialize search components
        # These will be created on first move when we know board size
        self.zobrist = None
        self.transposition_table = TranspositionTable(max_size=50000)
        
        # Search configuration
        self.default_search_depth = 2  # Look ahead: our move + opponent response
        self.time_buffer = 0.1  # Reserve 0.1 seconds for safety
        
        # Move history to prevent immediate repetition
        self.last_move = None
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], 
              current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move using minimax search with strategic heuristics.
        
        ===== ENHANCED DECISION-MAKING WITH MINIMAX =====
        
        STEP 1: QUICK STRATEGIC ASSESSMENT
           - Determine if we need emergency defense
           - Check if we can win immediately
           - Decide search depth based on time remaining
        
        STEP 2: MINIMAX SEARCH WITH ALPHA-BETA PRUNING
           - Search game tree to given depth
           - Evaluate positions after opponent responses
           - Prune branches that can't improve our position
           - Use transposition table to avoid re-computing positions
        
        STEP 3: FALLBACK TO HEURISTIC
           - If time is very limited, skip minimax
           - Use strategic move generation + evaluation
        
        WHY MINIMAX + HEURISTICS:
        - Minimax looks ahead to opponent responses
        - Heuristics guide which branches to explore first
        - Combined approach: smart search + domain knowledge
        - Time management: adaptive depth based on clock
        
        Args:
            board: Current board state
            rows, cols: Board dimensions
            score_cols: Scoring area columns
            current_player_time: Time remaining for us
            opponent_time: Time remaining for opponent
        
        Returns:
            Best move found by search
        """
        # Initialize Zobrist hasher on first call
        if self.zobrist is None:
            self.zobrist = ZobristHash(rows, cols)
        
        # ===== STEP 1: COMPREHENSIVE POSITION ANALYSIS =====
        # Single-pass analysis: generate all moves + evaluate position
        print(f"\n{'='*60}")
        print(f"[TURN START] Player: {self.player}")
        print(f"[TIME] Remaining: {current_player_time:.2f}s")
        
        analysis = analyze_position_with_moves(board, self.player, rows, cols, score_cols)
        
        print(f"[MOVES GENERATED]")
        print(f"  - Winning moves: {len(analysis['winning_moves'])}")
        print(f"  - Scoring moves: {len(analysis['scoring_moves'])}")
        print(f"  - Advancing moves: {len(analysis['advancing_moves'])}")
        print(f"  - Push moves: {len(analysis['push_moves'])}")
        print(f"  - Defensive moves: {len(analysis['defensive_moves'])}")
        print(f"  - Setup moves: {len(analysis['setup_moves'])}")
        
        print(f"[OPPONENT THREATS]")
        print(f"  - Threatening stones: {len(analysis['opponent_threats']['threatening_stones'])}")
        print(f"  - Rivers: {len(analysis['opponent_threats']['rivers'])}")
        print(f"  - Push threats: {len(analysis['opponent_threats']['push_threats'])}")
        
        print(f"[POSITION EVAL] Overall score: {analysis['overall_score']:.2f}")
        
        # If we can win immediately, just do it (no need to search)
        if analysis['winning_moves']:
            best_win = analysis['winning_moves'][0]
            print(f"[DECISION] WINNING MOVE: {best_win}")
            print(f"{'='*60}\n")
            return best_win
        
        # ===== STEP 2: DETERMINE STRATEGIC MODE =====
        # Use analysis to determine game situation
        strategic_mode = "BALANCED"
        
        if analysis['opponent_threats']['threatening_stones']:
            if analysis['material_eval']['opp_stones'] >= 3:
                strategic_mode = "DESPERATE_DEFENSE"
            else:
                strategic_mode = "DEFENSIVE"
        elif analysis['material_eval']['my_stones'] >= 3:
            strategic_mode = "AGGRESSIVE"
        elif analysis['material_eval']['my_stones'] > analysis['material_eval']['opp_stones']:
            strategic_mode = "PROTECT_LEAD"
        elif analysis['material_eval']['my_stones'] < analysis['material_eval']['opp_stones']:
            strategic_mode = "DEFENSIVE"
        
        print(f"[STRATEGY] Mode: {strategic_mode}")
        
        # ===== STEP 3: BUILD MOVE POOL BASED ON STRATEGY =====
        # Select move types based on strategic situation
        candidate_moves = []
        move_categories = []
        
        if strategic_mode == "DESPERATE_DEFENSE":
            # EMERGENCY: Only defensive moves
            candidate_moves.extend(analysis['defensive_moves'])
            move_categories.append(f"defensive({len(analysis['defensive_moves'])})")
            # Also try scoring if possible (best defense is offense)
            if analysis['scoring_moves']:
                candidate_moves.extend(analysis['scoring_moves'])
                move_categories.append(f"scoring({len(analysis['scoring_moves'])})")
        
        elif strategic_mode == "DEFENSIVE":
            # Focus on defense but include some offense
            candidate_moves.extend(analysis['defensive_moves'])
            move_categories.append(f"defensive({len(analysis['defensive_moves'])})")
            if analysis['scoring_moves']:
                candidate_moves.extend(analysis['scoring_moves'])
                move_categories.append(f"scoring({len(analysis['scoring_moves'])})")
            if analysis['advancing_moves']:
                candidate_moves.extend(analysis['advancing_moves'][:5])
                move_categories.append(f"advancing({len(analysis['advancing_moves'][:5])})")
        
        elif strategic_mode == "PROTECT_LEAD":
            # Mix of defense and careful offense
            if analysis['scoring_moves']:
                candidate_moves.extend(analysis['scoring_moves'])
                move_categories.append(f"scoring({len(analysis['scoring_moves'])})")
            candidate_moves.extend(analysis['defensive_moves'][:5])
            move_categories.append(f"defensive({len(analysis['defensive_moves'][:5])})")
            if analysis['push_moves']:
                candidate_moves.extend(analysis['push_moves'][:3])
                move_categories.append(f"push({len(analysis['push_moves'][:3])})")
        
        elif strategic_mode == "AGGRESSIVE":
            # Focus on offense
            if analysis['scoring_moves']:
                candidate_moves.extend(analysis['scoring_moves'])
                move_categories.append(f"scoring({len(analysis['scoring_moves'])})")
            if analysis['advancing_moves']:
                candidate_moves.extend(analysis['advancing_moves'][:8])
                move_categories.append(f"advancing({len(analysis['advancing_moves'][:8])})")
            if analysis['push_moves']:
                candidate_moves.extend(analysis['push_moves'][:5])
                move_categories.append(f"push({len(analysis['push_moves'][:5])})")
        
        else:  # BALANCED
            # Include all types for flexibility
            if analysis['scoring_moves']:
                candidate_moves.extend(analysis['scoring_moves'])
                move_categories.append(f"scoring({len(analysis['scoring_moves'])})")
            if analysis['advancing_moves']:
                candidate_moves.extend(analysis['advancing_moves'][:5])
                move_categories.append(f"advancing({len(analysis['advancing_moves'][:5])})")
            if analysis['push_moves']:
                candidate_moves.extend(analysis['push_moves'][:3])
                move_categories.append(f"push({len(analysis['push_moves'][:3])})")
            if analysis['defensive_moves']:
                candidate_moves.extend(analysis['defensive_moves'][:3])
                move_categories.append(f"defensive({len(analysis['defensive_moves'][:3])})")
        
        # ALWAYS include some setup moves for all strategies (rivers are important!)
        if analysis['setup_moves']:
            candidate_moves.extend(analysis['setup_moves'][:8])
            move_categories.append(f"setup({len(analysis['setup_moves'][:8])})")
        
        print(f"[MOVE POOL] {', '.join(move_categories)}")
        
        # ===== STEP 4: FALLBACK IF NO MOVES =====
        if not candidate_moves:
            print(f"[WARNING] No strategic moves! Fallback to all possible moves")
            candidate_moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        
        if not candidate_moves:
            print(f"[ERROR] No legal moves available!")
            print(f"{'='*60}\n")
            return None
        
        # ===== STEP 4.5: FILTER OUT IMMEDIATE REPETITIONS =====
        # Prevent flipping the same piece back and forth
        if self.last_move and self.last_move.get('action') == 'flip':
            last_pos = self.last_move.get('from')
            filtered_moves = []
            for move in candidate_moves:
                # Skip if it's a flip at the same position we just flipped
                if move.get('action') == 'flip' and move.get('from') == last_pos:
                    continue
                filtered_moves.append(move)
            
            if filtered_moves:  # Use filtered if we have alternatives
                print(f"[ANTI-REPEAT] Filtered out flip at {last_pos} (would undo last move)")
                candidate_moves = filtered_moves
        
        # ===== STEP 5: SELECT BEST MOVE BY SCORING =====
        # Evaluate each candidate by simulating and scoring resulting position
        best_move = None
        best_score = float('-inf')
        
        print(f"[EVALUATION] Scoring {len(candidate_moves)} moves...")
        for i, move in enumerate(candidate_moves):  # Top 15 to balance speed/quality
            try:
                # Simulate move
                success, result = simulate_move(board, move, self.player, rows, cols, score_cols)
                if success:
                    new_board = result
                    # Evaluate resulting position
                    score = evaluate_position(new_board, self.player, rows, cols, score_cols)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                    
                    # Show detailed scoring for first few moves
                    if i < 5:
                        move_summary = f"{move.get('action')}"
                        if move.get('from'):
                            move_summary += f" from {move['from']}"
                        if move.get('to'):
                            move_summary += f" to {move['to']}"
                        print(f"  [{i+1}] {move_summary} -> Score: {score:.2f}")
            except Exception as e:
                print(f"  [ERROR] Move {move} failed: {e}")
                continue
        
        if best_move:
            print(f"[DECISION] BEST MOVE: {best_move}")
            print(f"[SCORE] {best_score:.2f}")
        else:
            # Just pick first valid move
            best_move = candidate_moves[0]
            print(f"[DECISION] FALLBACK to first move: {best_move}")
        
        # Record this move to prevent immediate repetition
        self.last_move = best_move
        
        print(f"{'='*60}\n")
        return best_move

# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()