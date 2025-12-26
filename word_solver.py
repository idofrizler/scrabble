"""
Scrabble Word Solver

Finds the best possible word placement given a board state and rack tiles.
Uses the board's dictionary and scoring rules to find optimal moves.
"""

import time
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass


@dataclass
class Move:
    """Represents a possible word placement."""
    word: str
    row: int
    col: int
    horizontal: bool  # True = left-to-right, False = top-to-bottom
    score: int
    tiles_used: List[Tuple[int, int, str]]  # List of (row, col, letter) for new tiles
    
    def __repr__(self):
        direction = "H" if self.horizontal else "V"
        return f"Move({self.word}, ({self.row},{self.col}), {direction}, {self.score}pts)"


class WordSolver:
    """
    Finds optimal word placements for Scrabble.
    """
    
    # Standard Scrabble letter point values
    LETTER_VALUES = {
        'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1, 'L': 1, 'N': 1, 'S': 1, 'T': 1, 'R': 1,
        'D': 2, 'G': 2,
        'B': 3, 'C': 3, 'M': 3, 'P': 3,
        'F': 4, 'H': 4, 'V': 4, 'W': 4, 'Y': 4,
        'K': 5,
        'J': 8, 'X': 8,
        'Q': 10, 'Z': 10,
        '?': 0  # Blank tile
    }
    
    # Special cells
    SPECIAL_CELLS = {
        'TW': {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)},
        'DW': {(1, 1), (2, 2), (3, 3), (4, 4), (7, 7),
               (1, 13), (2, 12), (3, 11), (4, 10),
               (10, 4), (11, 3), (12, 2), (13, 1),
               (10, 10), (11, 11), (12, 12), (13, 13)},
        'TL': {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
               (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)},
        'DL': {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
               (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
               (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
               (12, 6), (12, 8), (14, 3), (14, 11)}
    }
    
    def __init__(self, dictionary: Set[str]):
        """
        Initialize solver with a dictionary of valid words.
        
        Args:
            dictionary: Set of valid words (uppercase)
        """
        self.dictionary = dictionary
        self.grid_size = 15
        
        # Build prefix set for fast pruning
        self.prefixes = set()
        for word in dictionary:
            for i in range(1, len(word) + 1):
                self.prefixes.add(word[:i])
    
    def find_best_move(self, board_state: List[List[Optional[Tuple[str, int]]]], 
                       rack_letters: str,
                       max_time_seconds: float = 5.0) -> Optional[Move]:
        """
        Find the highest-scoring valid move.
        
        Args:
            board_state: 15x15 grid with (letter, confidence) or None
            rack_letters: String of letters on rack (e.g., "ABCDEFG")
            max_time_seconds: Maximum time to search
            
        Returns:
            Best Move found, or None if no valid move exists
        """
        start_time = time.time()
        
        # Convert board to simple letter grid
        board = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if board_state[r][c] is not None:
                    board[r][c] = board_state[r][c][0]
        
        # Check if board is empty (first move)
        is_first_move = all(board[r][c] is None 
                          for r in range(self.grid_size) 
                          for c in range(self.grid_size))
        
        # Find anchor squares (empty cells adjacent to existing tiles, or center for first move)
        if is_first_move:
            anchors = [(7, 7)]  # Center only for first move
        else:
            anchors = self._find_anchors(board)
        
        if not anchors:
            return None
        
        # Convert rack to list of available letters
        rack = list(rack_letters.upper())
        
        # Find all valid moves
        best_move = None
        moves_found = 0
        
        for anchor_row, anchor_col in anchors:
            if time.time() - start_time > max_time_seconds:
                break
            
            # Try horizontal placements
            for move in self._find_moves_at_anchor(board, rack, anchor_row, anchor_col, 
                                                    horizontal=True, is_first_move=is_first_move):
                moves_found += 1
                if best_move is None or move.score > best_move.score:
                    best_move = move
            
            # Try vertical placements
            for move in self._find_moves_at_anchor(board, rack, anchor_row, anchor_col,
                                                    horizontal=False, is_first_move=is_first_move):
                moves_found += 1
                if best_move is None or move.score > best_move.score:
                    best_move = move
        
        elapsed = time.time() - start_time
        if best_move:
            print(f"Word solver: Found {moves_found} moves in {elapsed:.2f}s, best: {best_move}")
        else:
            print(f"Word solver: No valid moves found in {elapsed:.2f}s")
        
        return best_move
    
    def _find_anchors(self, board: List[List[Optional[str]]]) -> List[Tuple[int, int]]:
        """Find all anchor squares (empty cells adjacent to existing tiles)."""
        anchors = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if board[r][c] is None:
                    # Check if adjacent to an existing tile
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            if board[nr][nc] is not None:
                                anchors.append((r, c))
                                break
        return anchors
    
    def _find_moves_at_anchor(self, board: List[List[Optional[str]]], 
                               rack: List[str],
                               anchor_row: int, anchor_col: int,
                               horizontal: bool,
                               is_first_move: bool) -> List[Move]:
        """Find all valid moves that use the given anchor square."""
        moves = []
        
        # Direction vectors
        if horizontal:
            dr, dc = 0, 1
        else:
            dr, dc = 1, 0
        
        # Find left/up limit (how far back we can start the word)
        start_row, start_col = anchor_row, anchor_col
        limit = 0
        while True:
            prev_r = start_row - dr
            prev_c = start_col - dc
            if prev_r < 0 or prev_c < 0:
                break
            if board[prev_r][prev_c] is not None:
                # There's a tile here - extend through it
                start_row, start_col = prev_r, prev_c
            elif limit < len(rack):
                # Empty cell - we can potentially start here
                start_row, start_col = prev_r, prev_c
                limit += 1
            else:
                break
        
        # Try building words starting from each possible position
        for offset in range(limit + 1):
            word_start_r = start_row + offset * dr
            word_start_c = start_col + offset * dc
            
            # Generate words using DFS
            for move in self._generate_words(board, rack.copy(), 
                                             word_start_r, word_start_c,
                                             dr, dc, "", [], 
                                             anchor_row, anchor_col,
                                             is_first_move):
                moves.append(move)
        
        return moves
    
    def _generate_words(self, board: List[List[Optional[str]]],
                        rack: List[str],
                        row: int, col: int,
                        dr: int, dc: int,
                        current_word: str,
                        tiles_placed: List[Tuple[int, int, str]],
                        anchor_row: int, anchor_col: int,
                        is_first_move: bool) -> List[Move]:
        """Recursively generate valid words using DFS."""
        moves = []
        
        # Check bounds
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            # Check if current word is valid
            if len(current_word) >= 2 and current_word in self.dictionary:
                if tiles_placed:  # Must use at least one tile from rack
                    # Calculate word start position
                    start_r = row - len(current_word) * dr
                    start_c = col - len(current_word) * dc
                    
                    # Check that there's no tile immediately BEFORE the word start
                    prev_r = start_r - dr
                    prev_c = start_c - dc
                    if 0 <= prev_r < self.grid_size and 0 <= prev_c < self.grid_size:
                        if board[prev_r][prev_c] is not None:
                            # There's a tile before word start - this is an invalid placement
                            return moves
                    
                    # Verify anchor is covered
                    anchor_covered = any(r == anchor_row and c == anchor_col 
                                        for r, c, _ in tiles_placed)
                    # For first move, must cover center
                    if is_first_move:
                        covers_center = any(r == 7 and c == 7 for r, c, _ in tiles_placed) or \
                                       (any(r == 7 for r, c, _ in tiles_placed) and 
                                        any(c == 7 for r, c, _ in tiles_placed))
                        if not covers_center:
                            # Check if word passes through center
                            word_start_r = tiles_placed[0][0] if tiles_placed else row
                            word_start_c = tiles_placed[0][1] if tiles_placed else col
                            word_cells = []
                            for i, letter in enumerate(current_word):
                                cell_r = word_start_r + i * dr if dr else word_start_r
                                cell_c = word_start_c + i * dc if dc else word_start_c
                                word_cells.append((cell_r, cell_c))
                            if (7, 7) not in word_cells:
                                return moves
                    
                    if anchor_covered or (not is_first_move):
                        score = self._calculate_score(board, current_word, 
                                                     start_r, start_c,
                                                     dr, dc, tiles_placed)
                        horizontal = dc == 1
                        moves.append(Move(current_word, start_r, start_c, horizontal, score, tiles_placed.copy()))
            return moves
        
        # Check prefix validity for pruning
        if current_word and current_word not in self.prefixes:
            return moves
        
        # Check if current word is already valid
        if len(current_word) >= 2 and current_word in self.dictionary and tiles_placed:
            # Check if the next cell is empty (word can end here - no adjacent tile after)
            next_r, next_c = row, col
            next_empty = (next_r >= self.grid_size or next_c >= self.grid_size or 
                         board[next_r][next_c] is None)
            
            if next_empty:
                # Calculate word starting position
                start_r = row - len(current_word) * dr
                start_c = col - len(current_word) * dc
                
                # Check that there's no tile immediately BEFORE the word start
                # (which would make this an invalid partial word)
                prev_r = start_r - dr
                prev_c = start_c - dc
                prev_empty = (prev_r < 0 or prev_c < 0 or 
                             prev_r >= self.grid_size or prev_c >= self.grid_size or
                             board[prev_r][prev_c] is None)
                
                if not prev_empty:
                    # There's a tile before word start - skip this placement
                    pass
                else:
                    # Verify anchor is used (or first move covers center)
                    anchor_covered = any(r == anchor_row and c == anchor_col 
                                        for r, c, _ in tiles_placed)
                    
                    if is_first_move:
                        # Check if word passes through center
                        word_cells = [(start_r + i * dr, start_c + i * dc) 
                                     for i in range(len(current_word))]
                        covers_center = (7, 7) in word_cells
                        if covers_center:
                            score = self._calculate_score(board, current_word, start_r, start_c,
                                                         dr, dc, tiles_placed)
                            horizontal = dc == 1
                            moves.append(Move(current_word, start_r, start_c, horizontal, score, tiles_placed.copy()))
                    elif anchor_covered:
                        score = self._calculate_score(board, current_word, start_r, start_c,
                                                     dr, dc, tiles_placed)
                        horizontal = dc == 1
                        moves.append(Move(current_word, start_r, start_c, horizontal, score, tiles_placed.copy()))
        
        # Try extending the word
        if board[row][col] is not None:
            # Cell has a tile - must use it
            letter = board[row][col]
            new_word = current_word + letter
            moves.extend(self._generate_words(board, rack, row + dr, col + dc,
                                             dr, dc, new_word, tiles_placed,
                                             anchor_row, anchor_col, is_first_move))
        else:
            # Cell is empty - try each rack tile
            tried = set()
            for i, letter in enumerate(rack):
                if letter in tried:
                    continue
                tried.add(letter)
                
                new_rack = rack[:i] + rack[i+1:]
                new_word = current_word + letter
                new_tiles = tiles_placed + [(row, col, letter)]
                
                # Check cross-words validity
                if self._check_cross_word(board, row, col, letter, dr, dc):
                    moves.extend(self._generate_words(board, new_rack, row + dr, col + dc,
                                                     dr, dc, new_word, new_tiles,
                                                     anchor_row, anchor_col, is_first_move))
        
        return moves
    
    def _check_cross_word(self, board: List[List[Optional[str]]],
                          row: int, col: int, letter: str,
                          main_dr: int, main_dc: int) -> bool:
        """Check if placing a letter creates a valid cross-word."""
        # Cross direction is perpendicular to main direction
        cross_dr = 1 - main_dr  # If main is horizontal (0,1), cross is vertical (1,0)
        cross_dc = 1 - main_dc
        
        # Check if there are any tiles in the cross direction
        has_cross = False
        
        # Look backwards
        r, c = row - cross_dr, col - cross_dc
        while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            if board[r][c] is None:
                break
            has_cross = True
            r -= cross_dr
            c -= cross_dc
        
        # Look forwards
        r, c = row + cross_dr, col + cross_dc
        while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            if board[r][c] is None:
                break
            has_cross = True
            r += cross_dr
            c += cross_dc
        
        if not has_cross:
            return True  # No cross-word formed
        
        # Build the cross-word
        cross_word = ""
        
        # Go to start of cross-word
        r, c = row, col
        while True:
            prev_r, prev_c = r - cross_dr, c - cross_dc
            if prev_r < 0 or prev_c < 0:
                break
            if board[prev_r][prev_c] is None:
                break
            r, c = prev_r, prev_c
        
        # Build word forward
        while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            if r == row and c == col:
                cross_word += letter
            elif board[r][c] is not None:
                cross_word += board[r][c]
            else:
                break
            r += cross_dr
            c += cross_dc
        
        # Check if cross-word is valid
        return len(cross_word) < 2 or cross_word in self.dictionary
    
    def _calculate_score(self, board: List[List[Optional[str]]],
                         word: str, start_row: int, start_col: int,
                         dr: int, dc: int,
                         tiles_placed: List[Tuple[int, int, str]]) -> int:
        """Calculate the score for a word placement including cross-words."""
        tiles_placed_set = {(r, c) for r, c, _ in tiles_placed}
        tiles_placed_letters = {(r, c): letter for r, c, letter in tiles_placed}
        
        total_score = 0
        
        # --- Main word score ---
        word_score = 0
        word_multiplier = 1
        
        for i, letter in enumerate(word):
            r = start_row + i * dr
            c = start_col + i * dc
            
            letter_value = self.LETTER_VALUES.get(letter, 0)
            
            # Apply multipliers only to newly placed tiles
            if (r, c) in tiles_placed_set:
                if (r, c) in self.SPECIAL_CELLS['DL']:
                    letter_value *= 2
                elif (r, c) in self.SPECIAL_CELLS['TL']:
                    letter_value *= 3
                
                if (r, c) in self.SPECIAL_CELLS['DW']:
                    word_multiplier *= 2
                elif (r, c) in self.SPECIAL_CELLS['TW']:
                    word_multiplier *= 3
            
            word_score += letter_value
        
        total_score += word_score * word_multiplier
        
        # --- Cross-word scores ---
        # For each newly placed tile, calculate score of any perpendicular word formed
        cross_dr = 1 - dr  # Perpendicular direction
        cross_dc = 1 - dc
        
        for (r, c, placed_letter) in tiles_placed:
            cross_word_score = 0
            cross_word_multiplier = 1
            cross_word_length = 0
            
            # Find start of cross-word (go backwards)
            start_r, start_c = r, c
            while True:
                prev_r = start_r - cross_dr
                prev_c = start_c - cross_dc
                if prev_r < 0 or prev_c < 0 or prev_r >= self.grid_size or prev_c >= self.grid_size:
                    break
                if board[prev_r][prev_c] is None and (prev_r, prev_c) not in tiles_placed_letters:
                    break
                start_r, start_c = prev_r, prev_c
            
            # Calculate cross-word score
            cr, cc = start_r, start_c
            while cr < self.grid_size and cc < self.grid_size:
                if board[cr][cc] is not None:
                    letter = board[cr][cc]
                    letter_value = self.LETTER_VALUES.get(letter, 0)
                    cross_word_score += letter_value
                    cross_word_length += 1
                elif (cr, cc) in tiles_placed_letters:
                    letter = tiles_placed_letters[(cr, cc)]
                    letter_value = self.LETTER_VALUES.get(letter, 0)
                    
                    # Apply multipliers only to the newly placed tile
                    if (cr, cc) == (r, c):
                        if (r, c) in self.SPECIAL_CELLS['DL']:
                            letter_value *= 2
                        elif (r, c) in self.SPECIAL_CELLS['TL']:
                            letter_value *= 3
                        
                        if (r, c) in self.SPECIAL_CELLS['DW']:
                            cross_word_multiplier *= 2
                        elif (r, c) in self.SPECIAL_CELLS['TW']:
                            cross_word_multiplier *= 3
                    
                    cross_word_score += letter_value
                    cross_word_length += 1
                else:
                    break
                cr += cross_dr
                cc += cross_dc
            
            # Only count cross-words of length >= 2
            if cross_word_length >= 2:
                total_score += cross_word_score * cross_word_multiplier
        
        # Add bingo bonus
        if len(tiles_placed) == 7:
            total_score += 50
        
        return total_score
