using CocosSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace MindMove.Cocos
{
    public class MoveSimulator
    {
        public enum SimMode
        {
            Random,
            Targeted
        }

        const int TargetedRandomCount = 4; 
        const int TargetedLastAngleCount = 4;
        const int TargetedTargetScoreCount = 4;
        public enum TargetedModeOptions
        {
            Random,
            LastAngle,
            TargetScore
        }
        List<TargetedModeOptions> mTargetModeOptions;

        double mDistanceToRenewRadians; // This is to be used along with state virtualization

        readonly double mUpdateDistance;
        readonly float mTraceLineWidth;
        readonly DrawRect mBounds;

        readonly List<ScoredDraw> mScoredDraws;

        readonly bool mUseVirtualization = false;

        DrawPoint mLastPt;

        readonly Random mRnd;

        double mLastState = GameScene.StateNotInit;
        double mLastState2 = GameScene.StateNotInit;
        double mLastStateVirt = GameScene.StateNotInit;

        bool mNegativeCurve = false;

        double mPositiveTotal = 0;
        double mNegativeTotal = 0;
        double mSumRawScores = 0;

        bool mRenewOptimalRadians = true;
        bool mSkipNextRenew = false;
        double mCurveRadians = 0;
        double mLastOptimalRadians = 0;
        double mLatestRelativeAngle = 90;

        const int DirectionalGranularity = 90;

        public MoveSimulator(DrawRect bounds, float updateDistance, float traceLineWidth, bool useVirtualization = true)
        {
            mBounds = bounds;
            mUpdateDistance = updateDistance;
            mTraceLineWidth = traceLineWidth;
            mDistanceToRenewRadians = mUpdateDistance;

            mScoredDraws = new List<ScoredDraw>();
            mRnd = new Random();
            SetNegativeCurve();
            mLastState = GameScene.StateNotInit;

            mUseVirtualization = useVirtualization;
        }

        ~MoveSimulator()
        {
            if (mScoredDraws != null) { mScoredDraws.Clear(); }
        }

        public bool SimMove(SimMode mode, double distance, ref CCLayer layer, ref ShortTouchHistory history, ref MovePredictionEngineExt engine, ThemeColors themeColors = null)
        {
            double radians = GetOptimalRadians(mode, ref engine);

            DrawPoint prevPt;
            if (!history.GetLastTouchPoint(out prevPt))
            {
                prevPt = mBounds.Center; // Currently, drawing always starts in the center of the bounds.
            }

            float dx;
            float dy;
            mLastPt = prevPt;
            if (distance >= mUpdateDistance)
            {
                dx = (float)(distance * Math.Cos(radians));
                dy = (float)(distance * Math.Sin(radians));
            }
            else
            {
                GetNextRandomizedCurveRadians(mode);
                dx = (float)(distance * Math.Cos(radians + mCurveRadians));
                dy = (float)(distance * Math.Sin(radians + mCurveRadians));
            }
            mLastPt.X += dx;
            mLastPt.Y += dy;

            // Keep within bounds
            // Also turn the draw around, regardless of how this effects score, to keep it more in the center
            if (mLastPt.X > mBounds.MaxX || mLastPt.X < mBounds.MinX || mLastPt.Y > mBounds.MaxY || mLastPt.Y < mBounds.MinY)
            {
                // Make the radians head towards center
                mLastOptimalRadians = Math.Atan2(mLastPt.Y - mBounds.Center.Y, mLastPt.X - mBounds.Center.X) + Math.PI;
                mCurveRadians = 0;

                mSkipNextRenew = true; // We skip the next renewal, so we can get away from the edges of the bounds.

                dx = (float)(distance * Math.Cos(mLastOptimalRadians));
                dy = (float)(distance * Math.Sin(mLastOptimalRadians));

                mLastPt.X = prevPt.X + dx;
                mLastPt.Y = prevPt.Y + dy;
            }

            // Fail safe
            if (mLastPt.X > mBounds.MaxX) { mLastPt.X = mBounds.MaxX; }
            else if (mLastPt.X < mBounds.MinX) { mLastPt.X = mBounds.MinX; }
            if (mLastPt.Y > mBounds.MaxY) { mLastPt.Y = mBounds.MaxY; }
            else if (mLastPt.Y <= mBounds.MinY) { mLastPt.Y = mBounds.MinY; }

            return AddMove(mLastPt, ref layer, ref history, ref engine, themeColors, mode);
        }

        private bool AddMove(DrawPoint pt, ref CCLayer layer, ref ShortTouchHistory history, ref MovePredictionEngineExt engine, ThemeColors themeColors, SimMode mode)
        {
            bool ret = false;
            history.Add(pt);
            if (UpdateMovementState(history, engine, mode))
            {
                if (mode != SimMode.Targeted) { mRenewOptimalRadians = true; }
                else if (mDistanceToRenewRadians <= 0)
                {
                    mRenewOptimalRadians = true;
                    mDistanceToRenewRadians = mUpdateDistance + (mRnd.NextDouble() * mUpdateDistance);
                }
                ret = true;
            }
            AddScoredDraw(ref history, ref layer, engine.historyDepth, themeColors);
            return ret;
        }

        private double GetOptimalRadians(SimMode mode, ref MovePredictionEngineExt engine)
        {   
            if (mRenewOptimalRadians)
            {
                if (mSkipNextRenew)
                {
                    mSkipNextRenew = false;
                }
                else
                {
                    double optimalState = 0;
                    if (mode == SimMode.Random)
                    {
                        optimalState = mRnd.NextDouble() * BaseMoveWeights.CircularRange;
                    }
                    else
                    {
                        optimalState = GetTargetedState(ref engine);
                    }

                    mLastOptimalRadians = (2 * Math.PI * (double)(optimalState / BaseMoveWeights.CircularRange) - Math.PI);
                }
                mCurveRadians = 0;
                mRenewOptimalRadians = false;
            }

            return mLastOptimalRadians;
        }

        double GetTargetedState(ref MovePredictionEngineExt engine)
        {
            RefreshTargetModeOptions();

            double state = 0;

            int selectIndex = mRnd.Next(0, mTargetModeOptions.Count);
            if (mTargetModeOptions[selectIndex] == TargetedModeOptions.Random)
            {
                // NeutralRandomPct of the time, randomly, make a random move
                state = mRnd.NextDouble() * BaseMoveWeights.CircularRange;
            }
            else if (mLastState != GameScene.StateNotInit && mTargetModeOptions[selectIndex] == TargetedModeOptions.LastAngle)
            {
                // Repeat the last relative angle, either clockwise or counter clockwise (randomly)
                if (mRnd.Next(0, 2) == 0)
                {
                    state = mLastState + mLatestRelativeAngle;
                    if (state >= BaseMoveWeights.CircularRange) { state -= BaseMoveWeights.CircularRange; }
                }
                else
                {
                    state = mLastState - mLatestRelativeAngle;
                    if (state < 0) { state += BaseMoveWeights.CircularRange; }
                }
            }
            else if (mTargetModeOptions[selectIndex] == TargetedModeOptions.TargetScore)
            {
                // Otherwise, make a move that optimally brings the current score closest to 0
                // TODO: Virtualize the examined states, to get the truly optimal state? Still does a decent job of reaching target without it.
                List<double> dirScores = engine.ExamineDirectionalScores(DirectionalGranularity);
                if (dirScores != null)
                {
                    bool alternateMirror = mRnd.Next(0, 2) == 0; // This is to make it when state values are mirrored, it alternates which side to favor.
                    double optimalDiff = mPositiveTotal + mNegativeTotal;
                    for (int j = 0; j < dirScores.Count; j++)
                    {
                        double diffScore = Math.Abs(dirScores[j] + mSumRawScores); // TODO: This logic only works when target score = 0.
                        if (alternateMirror ? diffScore < optimalDiff : diffScore <= optimalDiff)
                        {
                            state = ((double)j * BaseMoveWeights.CircularRange) / (double)dirScores.Count;
                            optimalDiff = diffScore;
                        }
                    }
                }
            }
            mTargetModeOptions.RemoveAt(selectIndex);

            return state;
        }

        void RefreshTargetModeOptions()
        {
            if (mTargetModeOptions != null && mTargetModeOptions.Count > 0) { return; }

            if (mTargetModeOptions == null)
            {
                mTargetModeOptions = new List<TargetedModeOptions>();
            }
            else
            {
                mTargetModeOptions.Clear();
            }

            for (int j = 0; j < TargetedRandomCount; j++) { mTargetModeOptions.Add(TargetedModeOptions.Random); }
            for (int j = 0; j < TargetedLastAngleCount; j++) { mTargetModeOptions.Add(TargetedModeOptions.LastAngle); }
            for (int j = 0; j < TargetedTargetScoreCount; j++) { mTargetModeOptions.Add(TargetedModeOptions.TargetScore); }
        }

        private void GetNextRandomizedCurveRadians(SimMode mode)
        {
            const double CurveRadiansMult = 0.25;
            double curveRadiansMult = mRnd.NextDouble() * CurveRadiansMult;
            double rnd = mRnd.NextDouble() - 0.75; //1 - Math.Sqrt(mRnd.NextDouble());
            if (mNegativeCurve) { rnd = -rnd; }
            mCurveRadians += (rnd * curveRadiansMult) * Math.PI;
            if (mCurveRadians > Math.PI) { mCurveRadians = Math.PI; }
            if (mCurveRadians < -Math.PI) { mCurveRadians = -Math.PI; }
        }

        private bool UpdateMovementState(ShortTouchHistory history, MovePredictionEngineExt engine, SimMode mode)
        {
            bool ret = false;

            double distanceSinceLast = history.GetAggregateDistance();
            if (distanceSinceLast > mUpdateDistance)
            {
                // Get the direction state for this tile, and record it
                double absState;
                double degrees;
                history.GetDirectionState(out absState, out degrees, updateDistance: mUpdateDistance);

                double state = SharedParams.VirtualizeState(absState, ref mLastState, ref mLastState2, ref mLastStateVirt, ref mLatestRelativeAngle, useVirtualization: mUseVirtualization);

                ret = true;
                history.ReduceAggregateDistance(mUpdateDistance, mUpdateDistance);
                if (mode == SimMode.Targeted) { mDistanceToRenewRadians -= distanceSinceLast; }

                Move exp;
                engine.RecordMoveAndGetPredicted(state, out exp);
                double add, neg;
                double score = engine.GetScoringWeight(state, exp, out add, out neg);
                mPositiveTotal += add;
                mNegativeTotal += neg;
                mSumRawScores += score;

                SetNegativeCurve();
            }

            return ret;
        }

        public double GetRunningScore()
        {
            double score = 0;
            if (mPositiveTotal > 0 && mNegativeTotal > 0)
            {
                if (mPositiveTotal > mNegativeTotal)
                {
                    score = 2 - (mNegativeTotal / mPositiveTotal);
                }
                else
                {
                    score = mPositiveTotal / mNegativeTotal;
                }
            }

            return score;
        }

        public void ResetRunningScore()
        {
            mPositiveTotal = 0;
            mNegativeTotal = 0;
        }

        public int GetMoveCount()
        {
            if (mScoredDraws == null) { return 0; }
            return mScoredDraws.Count();
        }

        private void AddScoredDraw(ref ShortTouchHistory history, ref CCLayer layer, int historyDepth, ThemeColors themeColors)
        {
            List<DrawPoint> lastStatePath = history.GetLastDirStatePoints();

            if (lastStatePath != null)
            {
                // TODO: Don't copy this logic from GameScene::DrawTraceLine
                double relativeAngle = (mLatestRelativeAngle / 180.0);
                List<double> latestAvgRelAngle = new List<double>();
                latestAvgRelAngle.Add(relativeAngle);
                if (mScoredDraws.Count > 2)
                {
                    latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 1].relativeAngleRaw);
                    latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 2].relativeAngleRaw);
                    latestAvgRelAngle.Add(mScoredDraws[mScoredDraws.Count - 3].relativeAngleRaw);
                }
                else
                {
                    latestAvgRelAngle.Add(0.5);
                    latestAvgRelAngle.Add(0.5);
                    latestAvgRelAngle.Add(0.5);
                }
                double latestAvgRelativeAngle = latestAvgRelAngle.Average();

                DrawColor color;
                DrawColor color2;
                if (themeColors != null)
                {
                    color = SharedParams.GetColorPatternsColor(latestAvgRelativeAngle, themeColors); // TODO: Pass dualColors parameter
                    color2 = themeColors.AveragedColor;
                }
                else
                {
                    color = new DrawColor(128, 128, 128); // Grey
                    color2 = color;
                }

                mScoredDraws.Add(new ScoredDraw(ref layer, lastStatePath, color, 1.0, 0, mTraceLineWidth, latestAvgRelativeAngle, relativeAngle, color2));
                mScoredDraws[mScoredDraws.Count - 1].Draw();
                if (mScoredDraws.Count >= historyDepth)
                {
                    mScoredDraws[mScoredDraws.Count - historyDepth].MakeInvisible(true);
                    mScoredDraws.RemoveAt(mScoredDraws.Count - historyDepth);
                }
            }
        }

        public void ClearScoredDraws()
        {
            if (mScoredDraws != null)
            {
                try
                {
                    for (int j = 0; j < mScoredDraws.Count; j++)
                    {
                        mScoredDraws[j].MakeInvisible(true);
                    }
                }
                catch { }
                mScoredDraws.Clear();
            }
        }

        public List<ScoredDraw> GetScoredDraws()
        {
            return mScoredDraws; 
        }

        public DrawPoint GetLastPt()
        {
            return mLastPt;
        }

        public void GetRecentVirtalizedMoves(out double lastState, out double lastState2, out double lastVirtState, out double latestRelativeAngle)
        {
            lastState = mLastState;
            lastState2 = mLastState2;
            lastVirtState = mLastStateVirt;
            latestRelativeAngle = mLatestRelativeAngle;
        }

        private void SetNegativeCurve()
        {
            if (mRnd.Next(0, 2) == 0) { mNegativeCurve = true; }
            else { mNegativeCurve = false; }
        }
    }
}