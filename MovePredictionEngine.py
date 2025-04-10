import math
import random

"""
This engine predicts moves in a 360 degree range, based on best fit to the existing move sequence,
and scores moves by how much they align with, or diverge from, the predicted move. 
The result is simple, repetitive move sequences score low, while complex, varied sequences score high.

This is the scoring algorithm used in the drawing game 'Mind Unbind'. The original code is in C#, the logic and behavior here are equivilent. 

Its primary goals are:
    1. Minimal impact from heuristic logic and arbitrary parameters. 
    2. Scores match intuition about what move sequences are "varied" vs. "repetitive" (and to what extent).
       This is achieved by starting with intuitive structures, that are then processed with streamlined calculations.
       Ideally, a simple instruction like "vary your drawing" is sufficient to play, and attain high scores. 
    3. Possible scores are within a range, symmetrical around 'random movement', which scores in the middle of the range.   
    4. Fast performance enables real-time scoring and feedback. 

Summary of how it works:

- Initialize: set up 30 static Move structures, each having 30 states, with weight gradients from 1.0 to -1.0. 
  The weights are circularly symmetrical, and sum to 0. Each state's weight is 0.1333 different from its neighbors (0.1333 * 30 = 2.0).
  E.g. For state_1 (s1) = 1.0: s0 = 0.8667, s2 = 0.8667... s25 = 0.2, s7 = 0.2... s22 = -0.2, s10 = -0.2... s16 = -1.0.
  Note with 30 states, base directional granularity is 12 degrees.
  
- For each round (i.e. a single sequence of moves):
    - Set up a Move history, initially empty. 
    - Set up weights for a state transitions, initially all 0's. 
    
    - For each new Move:
        - Get the predicted Move, based on the full Move history and state transition weights.
        - Record the new Move, updating the state transition weights.
            - For continuous (infinite) directional granularity, 'rotate' all state weights, based on the granular direction. 
              E.g. If move direction = 24 degrees: state_2 = 1.0, state_3 = 0.867. 
              If direction = 30 degrees: state_2 = 0.933, state_3 = 0.933.
              If direction = 35.25 degrees: state_2 = 0.875, state_3 = 0.9917.
              The other 28 states are also 'rotated'. E.g. for direction = 30, state_1 = 0.8, state_4 = 0.8. 
        - Score the new Move by how close it was to the predicted. 
          Update positive, and negative, score sums used for final scoring.
        
    - Calculate the final score so it's between 0 and 200:
        100 * ((posSum < negSum) ? (posSum / negSum) : 2 - (negSum / posSum)).
      Note: When posSum == negSum, score = 100. Random moves average this. 
      
Notes
- An emergent property is that varying change in relative angle, between moves, is the key to high scores.
- It detects both local and global patterns of movement, by accounting for the entire history each move. 
"""
class Move:
    def __init__(self, num_states):
        self.states = [0.0] * num_states

class BaseMoveWeights:
    # The model is limited to 30 states is for performance reasons, as it's O(n^2). This is the one of two significant heuristics in this algorithm.
    #
    # Scoring tests for StateCount=30, vs. StateCount=60, over 1000 randomized 300-move games. 
    # - 30-State: Mean score = 100.17, StdDev = 7.13
    # - 60-State: Mean score = 100.25, StdDev = 7.13
    # - Mean difference: 1.037, StdDev = 0.778. 95% CI upper-bound = 2.56 (note scores fit a normal distribution)
    #
    # Testing 30-state vs. 120-state, over 100 randomized 300-move games: Mean difference = 1.191, StdDev = 0.87. 95% upper-bound CI = 2.9.
    # Testing 30-state vs. 240-state, over 60 randomized 300-move games: Mean difference = 1.112, StdDev = 0.832. 95% upper-bound CI = 2.74.
    #
    # SUMMARY:
    # - 95% of the time, a 30-state model will be within 3 points of the 'true' score (in 300-move games) attained by a many-state model. 
    #
    # For 1000 randomized 80-move games, the mean difference for 30 vs. 60 states = 1.86, StdDev = 1.48. 95% upper-bound CI = 4.76.
    #
    # Note the core loop calculations could be parallelized, making it faster, enabling a higher StateCount.
    StateCount = 30
    BaseDiffPerState = 4.0 / StateCount

    MoveWeights = []
    SumMoveWeights = 0

    # The 360 degree range is virtualized to a smooth (infinte continuous) gradient, from 0 to 360.0 (and wraps around at 360.0). 
    CircularRange = 360.0

    @staticmethod
    def get_index_at_offset(t, offset):
        x = t + offset
        return x - (BaseMoveWeights.StateCount * (x // BaseMoveWeights.StateCount))

    @staticmethod
    def initialize():
        for t in range(BaseMoveWeights.StateCount):
            move = Move(BaseMoveWeights.StateCount)

            for j in range(BaseMoveWeights.StateCount // 2 + 1):
                # The weight at t = 1, weight at opposite of t = -1. Then, a gradient in between.
                # It creates a circle of symmetrical state weights, which sum to 0.
                weight = 1 - (BaseMoveWeights.BaseDiffPerState * j)
                move.states[BaseMoveWeights.get_index_at_offset(t, j)] = weight
                move.states[BaseMoveWeights.get_index_at_offset(t, -j)] = weight

            BaseMoveWeights.MoveWeights.append(move)

        for j in range(BaseMoveWeights.StateCount):
            BaseMoveWeights.SumMoveWeights += abs(BaseMoveWeights.MoveWeights[0].states[j])

BaseMoveWeights.initialize()

# Engine for prediciting circular moves, and scoring actual moves based on the predicted.
class MovePredictionEngine:
    def __init__(self, history_depth_count):
        self.history_depth = history_depth_count
        self.history = []
        self.weights = []

        # This is the other significant heuristic. Scoring still works well without this, usually giving similar results.
        #
        # The issue is that earlier moves have a greater weight on the final score, because
        # (1) they get scored based on a short, simple history, and (2) they also impact scores of later moves. 
        # This weights scores by current move count, making final scores align better with visible variation.
        # Notably, sequences will score more similarly when traversed in reversed order, and also after changing early moves.
        #
        # historyDepth^scoreScaler = e (~2.72). For 300 moves, move #5 gets 1.33; #75 gets 2.13; #150 gets 2.41; #225 gets 2.58.
        self.scoreScaler = (1.0 / math.log(self.history_depth)); # In get_scoring_weight(), we use moveCount^scoreScaler as a multiplier. 
        self.ScoreScalerMiddle = math.pow(self.history_depth / 2.0, self.scoreScaler)

        for d in range(self.history_depth):
            self.weights.append([Move(BaseMoveWeights.StateCount) for _ in range(BaseMoveWeights.StateCount)])

    # Given the directional degrees (from 0 to 1.0) of a 'move', records a Move, and gets the 'predicted' Move.
    # Note the predicted Move need not have a smooth weight distribution, however it will be symmetrical.
    def record_move_and_get_predicted(self, degrees):
        move = self.get_move(degrees)
        pred = Move(BaseMoveWeights.StateCount)

        # This is the core loop that calculates the predicted Move, and updates weights based on the new Move. 
        # Note the value caching outside the inner-most operations is for performance only.
        for d in range(len(self.history)):
            hist_move = self.history[d]
            move_weights = self.weights[d]
            for i in range(BaseMoveWeights.StateCount):
                move_weights_i = move_weights[i]
                move_state_i = move.states[i]
                for j in range(BaseMoveWeights.StateCount):
                    pred.states[i] += hist_move.states[j] * move_weights_i.states[j]
                    move_weights_i.states[j] += (move_state_i * hist_move.states[j])

        self.history.insert(0, move)
        if len(self.history) > self.history_depth:
            self.history.pop()

        return pred

    # Get a score, given the actual 'move' direction, and the predicted Move.
    def get_scoring_weight(self, degrees, predicted):
        t, t1_weight, t2_weight = self.get_target_state_and_weights(degrees)

        neg_add = pos_add = 0
        score = 0
        sum_states = sum(abs(s) for s in predicted.states)
        if sum_states > 0:
            # Scores are typically between -1 and 1, though can be outside this range.
            score = (((predicted.states[t] * t1_weight) + (predicted.states[BaseMoveWeights.get_index_at_offset(t, 1)] * t2_weight)) / sum_states) * BaseMoveWeights.SumMoveWeights

            # Note that using just the raw score still works OK (i.e. score > 0 ? neg_add += score : pos_add -= score).
            # The below essentially makes the impact of 'draws' (e.g. scores near 0) equal to 'wins' and 'losses' (high or low scores).
            # It scales the total weight allocated (to neg_add and pos_add) based on the max possible score of the predicted Move. 
            # Note the predicted.states weights are symmetrical, so simply setting max to predicted.states.Max() should get the same result. 
            max_state = max(predicted.states)
            min_state = min(predicted.states)
            max_abs = max(max_state, abs(min_state))
            score_range_normalizer = (max_abs / sum_states) * BaseMoveWeights.SumMoveWeights 
            neg_add = (score_range_normalizer + score)
            pos_add = (score_range_normalizer - score)

            scaler = math.pow(len(self.history), self.scoreScaler)
            neg_add *= scaler
            pos_add *= scaler
            score *= scaler / self.ScoreScalerMiddle # The raw score is used for feedback purposes, and best to keep close to a (-1,1) range.

        return -score, pos_add, neg_add # Negative, so the caller can treat positive values as positive scores. 

    def get_move(self, degrees):
        # First get the Move set up with base state values
        t, t1_weight, t2_weight = self.get_target_state_and_weights(degrees)
        base_move = BaseMoveWeights.MoveWeights[t]

        # Now virtualize the weights to match the 360 degree range
        move = Move(BaseMoveWeights.StateCount)
        for j in range(1, BaseMoveWeights.StateCount + 1):
            move.states[BaseMoveWeights.get_index_at_offset(t, j)] = (base_move.states[BaseMoveWeights.get_index_at_offset(t, j)] * t1_weight) + (base_move.states[BaseMoveWeights.get_index_at_offset(t, j - 1)] * t2_weight)
        return move

    def get_target_state_and_weights(self, degrees):
        if degrees == BaseMoveWeights.CircularRange:
            degrees = 0
        t = int(math.floor(BaseMoveWeights.StateCount * (degrees / BaseMoveWeights.CircularRange)))
        t1_weight = 1 - (((t + 1) / BaseMoveWeights.StateCount) - (degrees / BaseMoveWeights.CircularRange))
        t2_weight = 1 - t1_weight
        return t, t1_weight, t2_weight

    @staticmethod
    def test_score_move_series(degrees):
        pos_sum = 0
        neg_sum = 0
        engine = MovePredictionEngine(len(degrees))

        for degree in degrees:
            exp = engine.record_move_and_get_predicted(degree)
            score, add, neg = engine.get_scoring_weight(degree, exp)
            pos_sum += add
            neg_sum += neg

        if pos_sum < neg_sum:
            final_score = pos_sum / neg_sum
        else:
            final_score = 2 - (neg_sum / pos_sum)
        return final_score * 100  # Range is 0 to 200. Scores are symmetrical around 100 (where pos_sum = neg_sum).

    # Test function that simulates a 'game' of moveCount random moves, and returns the final score.
    # The score average converges to 100 over multiple 'games'. 
    @staticmethod
    def test_score_random_moves(move_count=300):
        rnd = random.Random()
        degrees = [rnd.random() * BaseMoveWeights.CircularRange for _ in range(move_count)]
        return MovePredictionEngine.test_score_move_series(degrees)

'''
Future expansion:
    3D support -- Simple: use 3 concurrent MPE's, one for each axis. Full: a spherical version of MPE. 
    Complexity scorer -- Score a sequence in both directions, average both scores. 
    Similarity of two sequences -- based on similarty of weight values for each index.
'''

if __name__ == "__main__":
    
    # For visually displaying a move sequence in Python
    from PIL import Image, ImageDraw, ImageFont
    def calculate_color(angle):
        normalized_angle = abs(angle) % 360
        if normalized_angle > 180:
            normalized_angle = 360 - normalized_angle

        if normalized_angle <= 90:
            green = int(255 * (1 - (normalized_angle / 90)))
            blue = int(255 * (normalized_angle / 90))
            red = 0
        else:
            blue = int(255 * (1 - ((normalized_angle - 90) / 90)))
            red = int(255 * ((normalized_angle - 90) / 90))
            green = 0

        return (red, green, blue)

    def draw_moves(degrees_list, score, line_length=72, image_size=960):
        image = Image.new("RGB", (image_size, image_size), "grey")
        draw = ImageDraw.Draw(image)

        center = (image_size // 2, image_size // 2)
        current_point = center
        previous_degrees = None

        for degrees in degrees_list:
            radians = math.radians(degrees)
            end_point = (
                current_point[0] + line_length * math.cos(radians),
                current_point[1] + line_length * math.sin(radians)
            )

            if previous_degrees is not None:
                relative_angle = degrees - previous_degrees
                color = calculate_color(relative_angle)
            else:
                color = (0, 0, 0)  

            draw.line([current_point, end_point], fill=color, width=4)
            current_point = end_point
            previous_degrees = degrees

        font = ImageFont.truetype("arial.ttf", 32)
        draw.text((5, 5), f"Score: {score:.2f}", fill="black", font=font)
        image.show()

    # Demos with 30-move 'games'

    # A rough circle (moves ~30 degrees apart, clockwise)
    # Scores 3.9
    moves = [0, 30, 62, 89, 120, 147, 181, 210, 233, 270, 300, 330, 359, 29, 63, 95, 120, 151, 182, 211, 241, 270, 300, 325, 0, 29, 58, 90, 118, 146]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    # A spiral (add 3 degrees each move, clockwise)
    # Scores 64.5
    moves = [0, 3, 9, 18, 30, 45, 63, 84, 108, 135, 165, 198, 234, 273, 315, 0, 48, 99, 154, 214, 287, 353, 62, 134, 209, 287, 8, 92, 179, 269]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    # Roughly back-forth, then roughly straight (repeated 3x)
    # Scores 34.7
    moves = [270, 100, 280, 95, 285, 97, 90, 93, 95, 91, 90, 270, 93, 280, 101, 283, 280, 277, 280, 285, 105, 280, 102, 277, 98, 95, 90, 100, 97, 90]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    # Above, but change to a rough boxes for final 10 moves
    # Scores 78.6
    moves = [270, 100, 280, 95, 285, 97, 90, 93, 95, 91, 90, 270, 93, 280, 101, 283, 280, 277, 280, 10, 100, 185, 280, 15, 100, 185, 275, 0, 90, 180]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    # High variation in angle changes
    # Scores 121.9
    moves = [290, 330, 180, 240, 80, 90, 135, 135, 85, 230, 0, 125, 270, 100, 235, 40, 30, 75, 105, 0, 110, 210, 315, 345, 125, 150, 155, 280, 30, 50]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    # Higher variation in angle changes
    # Scores 158.8
    moves = [90, 110, 30, 200, 155, 135, 330, 175, 325, 330, 315, 350, 25, 220, 60, 300, 300, 50, 45, 15, 200, 100, 320, 120, 330, 150, 50, 300, 220, 120]
    score = MovePredictionEngine.test_score_move_series(moves)
    draw_moves(moves, score)

    '''
    # Average scores of 300 randomized games of 30 moves each
    count = 300
    avg = 0
    for g in range(count):
        avg += MovePredictionEngine.test_score_random_moves(30)
    print(avg / count) # Will be ~100
    '''