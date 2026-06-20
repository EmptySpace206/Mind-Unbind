using System;
using System.Collections.Generic;

using CocosSharp;

namespace MindMove.Cocos
{
   public class ShortTouchHistory
    {
        // TODO: Remove the PointTick, just use DrawPoint. Also make the mRecentPoints always clear after each GetDirectionState.
        struct PointTick
        {
            public DrawPoint pt;
            public long tick;

            public PointTick(DrawPoint pt_, long tick_)
            {
                pt = pt_;
                tick = tick_;
            }
        }

        List<PointTick> mRecentPoints;
        long mLastDirectionCheckTick = 0;
        const int cMaxPoints = 12; // Used for visual experience purposes

        double mLastDirState = 0;
        double mLastDirStateRadians = 0;

        double mAggregateDistance = 0;

        readonly List<DrawPoint> mLastDirStatePoints;
        bool mLastDirStatePointsValid = false;

        readonly bool mSnapMode = false;

        public ShortTouchHistory(bool snapMode = false)
        {
            mRecentPoints = new List<PointTick>();
            mLastDirectionCheckTick = DateTime.UtcNow.Ticks;
            mLastDirStatePoints = new List<DrawPoint>();
            mSnapMode = snapMode;
        }

        public void Add(DrawPoint pt, double minDistanceForAdd = 0)
        {
            if (mRecentPoints.Count > 0)
            {
                if (mSnapMode)
                {
                    DrawPoint ptAnchor = mRecentPoints[0].pt;
                    float distance = (float)Math.Sqrt((Math.Pow(pt.X - ptAnchor.X, 2) + Math.Pow(pt.Y - ptAnchor.Y, 2)));
                    mAggregateDistance = distance;
                    mRecentPoints.Add(new PointTick(pt, DateTime.UtcNow.Ticks));
                }
                else
                { 
                    DrawPoint pt2 = mRecentPoints[mRecentPoints.Count - 1].pt;
                    float distance = (float)Math.Sqrt((Math.Pow(pt.X - pt2.X, 2) + Math.Pow(pt.Y - pt2.Y, 2)));
                    if (distance >= minDistanceForAdd)
                    {
                        mRecentPoints.Add(new PointTick(pt, DateTime.UtcNow.Ticks));
                        mAggregateDistance += distance;
                    }
                }
            }
            else
            {
                mRecentPoints.Add(new PointTick(pt, DateTime.UtcNow.Ticks));
            }
        }

        public double GetAggregateDistance()
        {
            return mAggregateDistance;
        }

        public void ReduceAggregateDistance(double amount, double maxRemaining)
        {
            mAggregateDistance -= amount;
            if (mAggregateDistance < 0) { mAggregateDistance = 0;}
            if (mAggregateDistance > maxRemaining) { mAggregateDistance = maxRemaining; }
        }

        public void Clear()
        {
            mRecentPoints.Clear();
            mLastDirStatePoints.Clear();
            mLastDirStatePointsValid = false;
            mLastDirState = 0;
            mLastDirStateRadians = 0;
            mLastDirectionCheckTick = DateTime.UtcNow.Ticks;
            mAggregateDistance = 0;
        }

        bool SnapModeGetDirectionState(out double state, double updateDistance = 0)
        {
            if (mRecentPoints.Count < 2) // Shouldn't actually happen
            {
                state = mLastDirState;
                return false; 
            }

            PointTick pt0 = mRecentPoints[0];
            PointTick pt1 = mRecentPoints[mRecentPoints.Count - 1];

            // If we overshot the distance, insert fake point at the update distance. 
            double overDist = mAggregateDistance - updateDistance;
            if (updateDistance != 0 && overDist > 0)
            {
                // Calculate the location of the fake point, and set it as the last point
                float t = 1f - (float)(overDist / mAggregateDistance);
                DrawPoint fPt = new DrawPoint(
                    ((1 - t) * pt0.pt.X) + (t * pt1.pt.X),
                    ((1 - t) * pt0.pt.Y) + (t * pt1.pt.Y));
                pt1 = new PointTick(fPt, pt1.tick);
            }

            // Compute the angle between our first and last point.
            double radians = Math.Atan2(pt1.pt.Y - pt0.pt.Y, pt1.pt.X - pt0.pt.X) + Math.PI;

            state = (radians / 2) * (BaseMoveWeights.CircularRange / Math.PI);
            mLastDirState = state;
            mLastDirStateRadians = radians;

            // Keep only the last point (the start of the new move)
            mLastDirStatePointsValid = false;
            mRecentPoints.Insert(0,pt1);
            mRecentPoints.RemoveRange(1, mRecentPoints.Count - 1);

            mAggregateDistance = 0;

            mLastDirStatePoints.Clear();
            mLastDirStatePoints.Add(pt0.pt);
            mLastDirStatePoints.Add(pt1.pt);
            mLastDirStatePointsValid = true;

            return true;
        }

        public bool GetDirectionState(out double state, out double degreesDelta, double updateDistance = 0)
        {
            if (mSnapMode)
            {
                degreesDelta = 0;
                return SnapModeGetDirectionState(out state, updateDistance);
            }

            int startIndex = mRecentPoints.Count;
            for (int j = 0; j < mRecentPoints.Count; j++)
            {
                if (mRecentPoints[j].tick >= mLastDirectionCheckTick)
                {
                    startIndex = j;
                    break;
                }
            }

            if (startIndex > 0) { startIndex--; } // Start from the end point of the last update

            // Have at least 2 points to be valid. Shouldn't actually happen.
            if (startIndex >= (mRecentPoints.Count - 1))
            {
                state = mLastDirState;
                degreesDelta = 0;
                return false;
            }

            // If we overshot the update distance, insert a fake point at the update distance
            double overDist = mAggregateDistance - updateDistance;
            PointTick originalFinalPt = new PointTick(mRecentPoints[mRecentPoints.Count - 1].pt, 0); // Update the time later in this function
            bool addFakePt = false;
            if (updateDistance != 0 && overDist > 0)
            {
                addFakePt = true;

                PointTick pt0 = mRecentPoints[mRecentPoints.Count - 2];
                PointTick pt1 = mRecentPoints[mRecentPoints.Count - 1];

                // Calculate the location of the fake point, and create it
                float t =  1f - (float)(overDist / mAggregateDistance);
                DrawPoint fPt = new DrawPoint(
                    ((1 - t) * pt0.pt.X) + (t * pt1.pt.X),
                    ((1 - t) * pt0.pt.Y) + (t * pt1.pt.Y));
                PointTick fakePt1 = new PointTick(fPt, pt1.tick); 

                // Replace the end point with the fake point
                mRecentPoints.RemoveAt(mRecentPoints.Count - 1);
                mRecentPoints.Add(fakePt1);

                mAggregateDistance = updateDistance;
            }

            double x1 = mRecentPoints[startIndex].pt.X;
            double y1 = mRecentPoints[startIndex].pt.Y;

            // For smooth radians, we take the average radians between each point and the start, rather than the radians between the first and last.
            // This smoothes out cases where an arc is made since the last direction check.
            float avgX = 0;
            float avgY = 0;
            int counter = 0;
            double sumDegreesDiff = 0;
            double lastDegrees = -1;
            for (int j = startIndex + 1; j < mRecentPoints.Count; j++)
            {
                avgX += mRecentPoints[j].pt.X;
                avgY += mRecentPoints[j].pt.Y;
                double degrees = (Math.Atan2(mRecentPoints[j].pt.Y - mRecentPoints[j-1].pt.Y, mRecentPoints[j].pt.X - mRecentPoints[j-1].pt.X) + Math.PI) * 180 / Math.PI;
                if (lastDegrees != -1)
                {
                    double relDegrees = SharedParams.GetRelativeAngle(degrees, lastDegrees);
                    sumDegreesDiff = degrees > lastDegrees ? sumDegreesDiff + relDegrees : sumDegreesDiff - relDegrees;
                }
                lastDegrees = degrees;
                counter++;
            }
            sumDegreesDiff = Math.Abs(sumDegreesDiff);
            avgX /= counter;
            avgY /= counter;
            double radians = Math.Atan2(avgY - y1, avgX - x1) + Math.PI;

            double relLast = SharedParams.GetRelativeAngle(radians * 180 / Math.PI, mLastDirStateRadians * 180 / Math.PI);
            degreesDelta = Math.Max(sumDegreesDiff, relLast);

            //System.Diagnostics.Debug.WriteLine("lastRad: " + mLastDirStateRadians + ", degreesDelta: " + degreesDelta + ", sumDegrees: " + sumDegreesDiff + ", avgDegrees : " + sumDegreesDiff / counter);

            state = (radians / 2) * (BaseMoveWeights.CircularRange / Math.PI);
            mLastDirState = state;
            mLastDirStateRadians = radians;

            long now = DateTime.UtcNow.Ticks;
            mLastDirectionCheckTick = now;

            mLastDirStatePointsValid = false;
            mLastDirStatePoints.Clear();
            for (int j = startIndex; j < mRecentPoints.Count; j++)
            {
                mLastDirStatePoints.Add(mRecentPoints[j].pt);
            }

            // Shrink list to max points
            if (mRecentPoints.Count > cMaxPoints)
            {
                mRecentPoints = mRecentPoints.GetRange(mRecentPoints.Count - cMaxPoints, cMaxPoints);
            }

            if (addFakePt)
            {
                // Make the original final point part of the next move.
                Add(originalFinalPt.pt);
                originalFinalPt.tick = now + 1;
            }

            mLastDirStatePointsValid = true;

            return true;
        }

        public bool GetLastTouchPoint(out DrawPoint pt)
        {
            if (mRecentPoints.Count > 0)
            {
                pt = mRecentPoints[mRecentPoints.Count - 1].pt;
                return true;
            }
            else
            {
                pt = new DrawPoint(0, 0);
                return false;
            }
        }

        public DrawPoint GetSnapGridOriginPoint()
        {
            if (mRecentPoints.Count > 0)
            {
                return mRecentPoints[0].pt;
            }
            else
            {
                return new DrawPoint(0, 0);
            }
        }

        public List<DrawPoint> GetLastDirStatePoints()
        {
            if (mLastDirStatePointsValid)
            {
                mLastDirStatePointsValid = false;
                return mLastDirStatePoints;
            }
            else
            {
                return null;
            }
        }
    }
}