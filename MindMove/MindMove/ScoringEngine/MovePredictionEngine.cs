
using System;
using System.Collections.Generic;
using System.Linq;

/// This engine predicts moves in a 360 degree range, based on best fit to the existing move sequence,
/// and scores moves by how much they align with, or diverge from, the predicted move. 
/// The result is simple, repetitive move sequences score low, while complex, varied sequences score high.
/// 
/// This is the scoring algorithm used in the drawing game 'Mind Unbind'. 
/// 
/// Its primary goals are:
///     1. Minimal impact from heuristic logic and arbitrary parameters. 
///     2. Scores match intuition about what move sequences are "varied" vs. "repetitive" (and to what extent).
///        This is achieved by starting with intuitive structures, that are then processed with streamlined calculations.
///        Ideally, a simple instruction like "vary your drawing" is sufficient to play, and attain high scores. 
///     3. Possible scores are within a range, symmetrical around 'random movement', which scores in the middle of the range.   
///     4. Fast performance enables real-time scoring and feedback. 
/// 
/// Summary of how it works:
/// 
/// - Initialize: set up 30 static Move structures, each having 30 states, with weight gradients from 1.0 to -1.0. 
///   The weights are circularly symmetrical, and sum to 0. Each state's weight is 0.1333 different from its neighbors (0.1333 * 30 = 2.0).
///   E.g. For state_1 (s1) = 1.0: s0 = 0.8667, s2 = 0.8667... s25 = 0.2, s7 = 0.2... s22 = -0.2, s10 = -0.2... s16 = -1.0.
///   Note with 30 states, base directional granularity is 12 degrees.
///   
/// - For each round (i.e. a single sequence of noves):
///     - Set up a Move history, initially empty. 
///     - Set up weights for a state transitions, initially all 0's. 
///     
///     - For each new Move:
///         - Get the predicted Move, based on the full Move history and state transition weights.
///         - Record the new Move, updating the state transition weights.
///             - For continuous (infinte) directional granularity, 'rotate' all state weights, based on the granular direction. 
///               E.g. If move direction = 24 degrees: state_2 = 1.0, state_3 = 0.867. 
///               If direction = 30 degrees: state_2 = 0.933, state_3 = 0.933.
///               If direction = 35.25 degrees: state_2 = 0.875, state_3 = 0.9917.
///               The other 28 states are also 'rotated'. E.g. for direction = 30, state_1 = 0.8, state_4 = 0.8. 
///         - Score the new Move by how close it was to the predicted. 
///           Update positive, and negative, score sums used for final scoring.
///         
///     - Calculate the final score so it's between 0 and 200:
///         100 * ((posSum < negSum) ? (posSum / negSum) : 2 - (negSum / posSum)).
///       Note: When posSum == negSum, score = 100. Random moves average this. 
///       
/// Notes
/// - An emergent property is that varying change in relative angle, between moves, is the key to high scores.
/// - It detects both local and global patterns of movement, by accounting for the entire history each move. 

namespace MindMove
{
    public struct Move
    {
        public double[] states;
        public Move(int numStates) { states = new double[numStates]; } // Auto-initializes to 0
    }

    // Sets up base weights for a circular, N-state structure. Runs on app startup. 
    public static class BaseMoveWeights
    {
        // The model is limited to 30 states is for performance reasons, as it's O(n^2). This is the one of two significant heuristics in this algorithm.
        //
        // Scoring tests for StateCount=30, vs. StateCount=60, over 1000 randomized 300-move games. 
        // - 30-State: Mean score = 100.17, StdDev = 7.13
        // - 60-State: Mean score = 100.25, StdDev = 7.13
        // - Mean difference: 1.037, StdDev = 0.778. 95% CI upper-bound = 2.56 (note scores fit a normal distribution)
        //
        // Testing 30-state vs. 120-state, over 100 randomized 300-move games: Mean difference = 1.191, StdDev = 0.87. 95% upper-bound CI = 2.9.
        // Testing 30-state vs. 240-state, over 60 randomized 300-move games: Mean difference = 1.112, StdDev = 0.832. 95% upper-bound CI = 2.74.
        //
        // SUMMARY:
        // - 95% of the time, a 30-state model will be within 3 points of the 'true' score (in 300-move games) attained by a many-state model. 
        //
        // For 1000 randomized 80-move games, the mean difference for 30 vs. 60 states = 1.86, StdDev = 1.48. 95% upper-bound CI = 4.76.
        //
        // Note the core loop calculations could be parallelized, making it faster, enabling a higher StateCount.
        public const int StateCount = 30;  
        public const double BaseDiffPerState = 4.0 / StateCount; 

        public static readonly List<Move> MoveWeights = new List<Move>();
        public static readonly double SumMoveWeights = 0;

        // The 360 degree range is virtualized to a smooth (infinte continuous) gradient, from 0 to 360.0 (and wraps around at 360.0). 
        public const double CircularRange = 360.0; 

        public static int GetIndexAtOffset(int t, int offset)
        {
            int x = t + offset;
            return x - (StateCount * (int)Math.Floor((double)x / StateCount));
        }

        static BaseMoveWeights()
        {
            for (int t = 0; t < StateCount; t++)
            {
                Move move = new Move(StateCount);
                for (int j = 0; j <= StateCount / 2; j++)
                {
                    // The weight at t = 1, weight at opposite of t = -1. Then, a gradient in between.
                    // It creates a circle of symmetrical state weights, which sum to 0.
                    double weight = 1 - (BaseDiffPerState * j);
                    move.states[GetIndexAtOffset(t, j)] = weight;
                    move.states[GetIndexAtOffset(t, -j)] = weight;
                }
                MoveWeights.Add(move);
            }
            for (int j = 0; j < StateCount; j++) { SumMoveWeights += Math.Abs(MoveWeights[0].states[j]); }
        }
    }

    // Engine for prediciting circular moves, and scoring actual moves based on the predicted.
    public class MovePredictionEngine
    {
        protected List<Move> history;
        protected List<Move[]> weights;
        public readonly int historyDepth; // I.e. the number of 'moves'

        readonly double scoreScaler;
        readonly double scoreScalerMiddle;

        public MovePredictionEngine(int historyDepthCount)
        {
            historyDepth = historyDepthCount;
            history = new List<Move>();
            weights = new List<Move[]>();
            for (int d = 0; d < historyDepth; d++)
            {
                weights.Add(new Move[BaseMoveWeights.StateCount]);
                for (int i = 0; i < BaseMoveWeights.StateCount; i++) { weights[d][i] = new Move(BaseMoveWeights.StateCount); }
            }

            // This is the other significant heuristic. Scoring still works well without this, usually giving similar results.
            //
            // The issue is that earlier moves have a greater weight on the final score, because
            // (1) they get scored based on a short, simple history, and (2) they also impact scores of later moves. 
            // This weights scores by current move count, making final scores align better with visible variation.
            //
            // historyDepth^scoreScaler = e (~2.72). For 300 moves, move #5 gets 1.33; #75 gets 2.13; #150 gets 2.41; #225 gets 2.58.
            scoreScaler = (1.0 / Math.Log(historyDepth)); // In GetScoringWeight(), we use moveCount^scoreScaler as a multiplier. 
            scoreScalerMiddle = Math.Pow(historyDepth / 2.0, scoreScaler);
        }
        ~MovePredictionEngine() { history.Clear(); weights.Clear(); }

        // Given the directional degrees (from 0 to 1.0) of a 'move', records a Move, and gets the 'predicted' Move.
        // Note the predicted Move need not have a smooth weight distribution, however it will be symmetrical.
        public virtual void RecordMoveAndGetPredicted(double degrees, out Move pred)
        {
            Move move = GetMove(degrees);
            pred = new Move(BaseMoveWeights.StateCount);

            // This is the core loop that calculates the predicted Move, and updates weights based on the new Move. 
            // Note the value cacheing outside the inner-most operations is for performance only.
            for (int d = 0; d < history.Count; d++)
            {
                Move histMove = history[d]; 
                Move histMoveReverse = history[(history.Count - 1) - d]; 
                Move[] moveWeights = weights[d];
                for (int i = 0; i < BaseMoveWeights.StateCount; i++)
                {
                    Move moveWeightsI = moveWeights[i];
                    double moveStateI = move.states[i];
                    for (int j = 0; j < BaseMoveWeights.StateCount; j++)
                    {
                        pred.states[i] += histMove.states[j] * moveWeightsI.states[j];
                        moveWeightsI.states[j] += (moveStateI * histMove.states[j]) + (moveStateI * histMoveReverse.states[j]);
                    }
                }
            }

            history.Insert(0, move);
            if (history.Count > historyDepth) { history.RemoveAt(history.Count - 1); }
        }

        // Get a score, given the actual 'move' direction, and the predicted Move.
        public double GetScoringWeight(double degrees, Move predicted, out double posAdd, out double negAdd)
        {
            int t = GetTargetStateAndWeights(degrees, out double t1Weight, out double t2Weight);

            negAdd = posAdd = 0;
            double score = 0;
            double sum = 0;
            foreach (double s in predicted.states) { sum += Math.Abs(s); }
            if (sum > 0)
            {
                // Scores are typically between -1 and 1, though can be outside this range.
                score = (((predicted.states[t] * t1Weight) + (predicted.states[BaseMoveWeights.GetIndexAtOffset(t, 1)] * t2Weight)) / sum) * BaseMoveWeights.SumMoveWeights;

                // Note that using just the raw score still works OK (i.e. score > 0 ? negAdd += score : posAdd -= score).
                // The below essentially makes the impact of 'draws' (e.g. scores near 0) equal to 'wins' and 'losses' (high or low scores).
                // It scales the total weight allocated (to negAdd and posAdd) based on the max possible score of the predicted Move. 
                // Note the predicted.states weights are symmetrical, so simply setting max to predicted.states.Max() should get the same result. 
                double max = Math.Max(predicted.states.Max(), Math.Abs(predicted.states.Min()));
                double scoreRangeNormalizer = (max / sum) * BaseMoveWeights.SumMoveWeights;
                negAdd = (scoreRangeNormalizer + score);
                posAdd = (scoreRangeNormalizer - score);

                double scaler = Math.Pow(history.Count, scoreScaler);
                posAdd *= scaler;
                negAdd *= scaler;
                score *= scaler / scoreScalerMiddle; // The raw score is used for feedback purposes, and best to keep close to a(-1, 1) range.
            }

            return -score; // Negative, so the caller can treat positive values as positive scores. 
        }

        protected Move GetMove(double degrees)
        {
            // First get the Move set up with base state values
            int t = GetTargetStateAndWeights(degrees, out double t1Weight, out double t2Weight);
            Move baseMove = BaseMoveWeights.MoveWeights[t];

            // Now virtualize the weights to match the 360 degree range
            Move move = new Move(BaseMoveWeights.StateCount);
            for (int j = 1; j <= BaseMoveWeights.StateCount; j++)
            {
                move.states[BaseMoveWeights.GetIndexAtOffset(t, j)] = (baseMove.states[BaseMoveWeights.GetIndexAtOffset(t, j)] * t1Weight) + (baseMove.states[BaseMoveWeights.GetIndexAtOffset(t, j - 1)] * t2Weight);
            }
            return move;
        }

        private int GetTargetStateAndWeights(double degrees, out double t1Weight, out double t2Weight)
        {
            if (degrees == BaseMoveWeights.CircularRange) { degrees = 0; } // Wrap around from 360 to 0
            int t = (int)Math.Floor(BaseMoveWeights.StateCount * (degrees / BaseMoveWeights.CircularRange));
            t1Weight = 1 - (((double)(t+1) / (double)BaseMoveWeights.StateCount) - (degrees / BaseMoveWeights.CircularRange)); 
            t2Weight = 1 - t1Weight;
            return t;
        }

        public static double Test_ScoreMoveSeries(List<double> degrees)
        {
            double posSum = 0;
            double negSum = 0;
            var engine = new MovePredictionEngine(degrees.Count);

            for (int j = 0; j < degrees.Count; j++)
            {
                engine.RecordMoveAndGetPredicted(degrees[j], out Move exp);
                double score = engine.GetScoringWeight(degrees[j], exp, out double add, out double neg);
                posSum += add;
                negSum += neg;
            }

            double finalScore;
            if (posSum < negSum) { finalScore = posSum / negSum; }
            else { finalScore = 2 - (negSum / posSum); }
            return finalScore * 100; // Range is 0 to 200. Scores are symmetrical around 100 (where posSum = negSum). 
        }

        // Test function that simulates a 'game' of moveCount random moves, and returns the final score.
        // The score average converges to 100 over multiple 'games'. 
        public static double Test_ScoreRandomMoves(int moveCount = 300)
        {
            Random rnd = new Random();
            List<double> degrees = new List<double>();
            for (int i = 0; i < moveCount; i++) { degrees.Add(rnd.NextDouble() * BaseMoveWeights.CircularRange); }
            return Test_ScoreMoveSeries(degrees);
        }

        // Returns a list of clockwise scores, starting from 0 degrees, for the next move.
        // Useful for generating moves that aim for a target score.
        public List<double> ExamineDirectionalScores(int granularityDegrees = 120)
        {
            Move predicted = new Move(BaseMoveWeights.StateCount);
            for (int d = 0; d < history.Count; d++)
            {
                Move histMove = history[d];
                Move[] moveWeights = weights[d];
                for (int i = 0; i < BaseMoveWeights.StateCount; i++)
                {
                    Move moveWeightsI = moveWeights[i];
                    for (int j = 0; j < BaseMoveWeights.StateCount; j++)
                    {
                        predicted.states[i] += histMove.states[j] * moveWeightsI.states[j];
                    }
                }
            }

            double increment = BaseMoveWeights.CircularRange / granularityDegrees;
            List<double> clockwiseScores = null;
            double sum = 0;
            foreach (double s in predicted.states) { sum += Math.Abs(s); }
            if (sum > 0)
            {
                clockwiseScores = new List<double>();
                for (double degrees = 0; degrees < BaseMoveWeights.CircularRange; degrees += increment)
                {
                    int t = GetTargetStateAndWeights(degrees, out double t1Weight, out double t2Weight);
                    double score = (((predicted.states[t] * t1Weight) + (predicted.states[BaseMoveWeights.GetIndexAtOffset(t, 1)] * t2Weight)) / sum) * BaseMoveWeights.SumMoveWeights;
                    clockwiseScores.Add(-score); // Negative, so the caller can treat positive values as positive scores.
                }
            }
            return clockwiseScores;
        }

        /// Future expansion:
        /// 3D support -- Simple: use 3 concurrent MPE's, one for each axis. Full: a spherical version of MPE. 
        /// Complexity scorer -- Score a sequence in both directions, average both scores.
        /// Similarity of two sequences -- based on similarty of weight values for each index.
    }
}
