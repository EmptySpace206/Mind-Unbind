using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;

namespace MindMove
{
    public class MovePredictionEngineExt : MovePredictionEngine
    {
        public const int StateCount = BaseMoveWeights.StateCount;

        readonly List<double> degreesHistory = new List<double>();

        public MovePredictionEngineExt(int historyDepthCount) : base(historyDepthCount)
        {
        }

        // This will get the contributing weight, from -1 to 1, for all moves contributing to the movesOfInterest.
        // Or if movesOfInterest is null, then it just gets contributing weights of all moves to all succeeding moves.
        // If endOfSearchIndex is specified, that index of move will be the final move for which is gathers contributions, and then it exits. 
        public List<double> GetContributingWeights(List<int> movesOfInterest)
        {
            List<int> movesToCont = new List<int>(movesOfInterest);

            // Set up the contributing weight for every move in the history, initialized to 0
            List<List<double>> contWeights = new List<List<double>>();
            for (int d = 0; d < historyDepth; d++)
            {
                contWeights.Add(new List<double>());
            }

            // Create a temporary weights, and history, just for this function call
            List<Move> historyTemp = new List<Move>();
            List<Move[]> weightsTemp = new List<Move[]>();
            for (int d = 0; d < historyDepth; d++)
            {
                weightsTemp.Add(new Move[StateCount]);
                for (int i = 0; i < StateCount; i++)
                {
                    weightsTemp[d][i] = new Move(StateCount);
                }
            }

            // Create reversed historical move lists, to make them in order of most recent to least recent
            List<Move> moveHist = new List<Move>();
            for (int d = history.Count - 1; d >= 0; d--)
            {
                moveHist.Add(history[d]);
            }

            // We need to re-play the entire history in order to get accurate contributing weights.
            // We do this again here, duplicating some code, in order to avoid impacting performance during the actual game.
            for (int d0 = 0; d0 < moveHist.Count; d0++) // We need to see the contribution of every move to every future move
            {
                Move move = moveHist[d0];
                List<Move> contMoves = new List<Move>();

                for (int d = 0; d < d0; d++)
                {
                    Move cm = new Move(StateCount);

                    // Perf optimization
                    Move histMove = historyTemp[d];
                    Move histMoveReverse = history[(d0 - 1) - d];
                    Move[] moveWeights = weightsTemp[d];

                    for (int i = 0; i < StateCount; i++)
                    {
                        // Perf optimization
                        Move moveWeightsI = moveWeights[i];
                        double moveStateI = move.states[i];

                        for (int j = 0; j < StateCount; j++)
                        {
                            cm.states[i] += histMove.states[j] * moveWeightsI.states[j];
                            moveWeightsI.states[j] += (moveStateI * histMove.states[j]) + (moveStateI * histMoveReverse.states[j]);
                        }
                    }

                    contMoves.Insert(0, cm);
                }

                // Add the weight contributed to move d0, for each move so far
                if (movesToCont.Contains(d0))
                {
                    movesToCont.Remove(d0);

                    // Get the original discreet move for d0. TODO: Shouldn't need to pass degreesHistory as a parameter
                    double result = degreesHistory[d0];

                    // Get the contributing weight of d to d0. 
                    for (int d = 0; d < contMoves.Count; d++)
                    {
                        double weight = GetScoringWeight(result, contMoves[d], out double add, out double neg);
                        contWeights[d].Add(weight);
                    }
                }

                historyTemp.Insert(0, move);

                if (movesToCont.Count == 0)
                {
                    break;
                }
            }

            // Normalize contributing moves between -1 and 1, and combine them across both models. 
            List<double> retWeights = new List<double>();
            for (int d = 0; d < historyDepth; d++)
            {
                if (contWeights[d].Count > 0)
                {
                    retWeights.Add(contWeights[d].Sum());
                }
                else
                {
                    retWeights.Add(0);
                }
            }          

            return retWeights;
        }

        // Used only in Continuity, for recording moves from prior game.
        // Faster because we don't fill out the expected move.
        public void RecordMove(double degrees)
        {
            degreesHistory.Add(degrees);
            if (degreesHistory.Count > historyDepth)
            {
                degreesHistory.RemoveAt(0);
            }

            Move move = GetMove(degrees);

            for (int d = 0; d < history.Count; d++)
            {
                // Perf optimization
                Move histMove = history[d];
                Move histMoveReverse = history[(history.Count - 1) - d];
                Move[] moveWeights = weights[d];

                for (int i = 0; i < StateCount; i++)
                {
                    // Perf optimization
                    Move moveWeightsI = moveWeights[i];
                    double moveStateI = move.states[i];

                    for (int j = 0; j < StateCount; j++)
                    {
                        moveWeightsI.states[j] += (moveStateI * histMove.states[j]) + (moveStateI * histMoveReverse.states[j]);
                    }
                }
            }

            history.Insert(0, move);
        }

        public override void RecordMoveAndGetPredicted(double degrees, out Move pred)
        {
            degreesHistory.Add(degrees);
            if (degreesHistory.Count > historyDepth)
            {
                degreesHistory.RemoveAt(0);
            }
            base.RecordMoveAndGetPredicted (degrees, out pred);
        }

        public List<double> GetDegreesHistory()
        {
            return degreesHistory;
        }
    }
}
