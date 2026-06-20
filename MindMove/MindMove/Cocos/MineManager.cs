using System;
using System.Collections.Generic;

using CocosSharp;
using static MindMove.Cocos.GameScene;

namespace MindMove.Cocos
{
    class MineManager
    {
        const double MinBaseUpdateDistance = 1.0 / 11.0;
        const double MineRadiusBase = (0.5 / 11.0) / MinBaseUpdateDistance; // For a square screen, we get 11x11 mines at default size.
        const double MineVoidStormRadiusBase = MineRadiusBase / 2.0;
        const double MineSmallRadiusBase = MineRadiusBase * 0.775; // Makes the area 0.6
        double mMineRadiusMiddle;

        const double VoidStormMineSizeRandomizer = 0.6;
        readonly List<double> mVoidStormMineSizeRandomizers;
        const double IncVoidStormMineSizePerRandomizer = 0.03;

        DrawColor mMineDormantColor;
        DrawColor mMineActiveColor;
        DrawColor mMineEmergeColor;
        DrawColor mMineBackgroundColor;

        DrawColor mMineXColorDark;
        DrawColor mMineXColorBright;
        DrawColor mMineActiveXColorDark;
        DrawColor mMineActiveXColorBright;

        readonly List<DrawPoint> mMineCenters;
        readonly List<double> mMineRadiuses;
        readonly List<DrawNode> mMines;
        readonly List<bool> mHitMines;

        double mMineRadiusMult = 1;

        bool mLastHitTestResult = false;

        const float MinDistFromFingerToSpawnBase = 1.5f;
        float MinDistFromFingerToSpawn;
        readonly float MinMineRadiusesToSpawn = 5f;
        const float MinDistFromFingerToSpawnMaxBase = 2.25f;
        float MinDistFromFingerMax;

        long mLastMineHitTicks = 0;
        const long MineTicksBetweeMineHitAndDeath = 500 * TimeSpan.TicksPerMillisecond;
        const long MineTicksDeathThresholdReductionPerSqrtMineCountXRadius = 100 * TimeSpan.TicksPerMillisecond;

        ThemeColors mThemeColors;
        readonly CCLayer mLayer;
        readonly PlayMode mPlayMode;

        float MineMinX;
        float MineMaxX;
        float MineMinY;
        float MineMaxY;

        bool mDisableMines = false;

        const int RadiusEmergeSteps = 16;
        readonly List<int> mEmergeStates;

        double mMineBaseSize;

        bool mSkipRemoveActive = false;

        public MineManager(CCLayer layer, ThemeColors colors, PlayMode playMode, bool v2Colors = true)
        {
            mLayer = layer;

            SetThemeColor(colors, v2Colors);

            mPlayMode = playMode;

            mMineCenters = new List<DrawPoint>();
            mMineRadiuses = new List<double>();
            mMines = new List<DrawNode>();
            mHitMines = new List<bool>();
            mEmergeStates = new List<int>();
            mVoidStormMineSizeRandomizers = new List<double>();
        }

        ~MineManager()
        {
            mMineCenters.Clear();
            mMineRadiuses.Clear();
            mMines.Clear();
            mHitMines.Clear();
            mEmergeStates.Clear();
            mVoidStormMineSizeRandomizers.Clear();
        }

        public void SetThemeColor(ThemeColors colors, bool v2Colors = false)
        {
            mThemeColors = colors;

            if (v2Colors)
            {
                mMineDormantColor = ThemeColors.GetColorForTargetLuminance(mThemeColors.MiddleColor, 0.5, false);
                mMineActiveColor = ThemeColors.GetColorForTargetLuminance(ThemeColors.InvertColor(mThemeColors.MiddleColor), 0.76, false);
                mMineEmergeColor = ThemeColors.GetColorForTargetLuminance(mThemeColors.MiddleColor, 0.68, false);
                mMineBackgroundColor = ThemeColors.AdjustColorsForMultipler(0.333, mThemeColors.AveragedColor);
            }
            else
            {
                mMineDormantColor = ThemeColors.GetColorForTargetLuminance(ThemeColors.AverageColors(mThemeColors.DistinctColorA, mThemeColors.DistinctColorB), 0.5, false);
                mMineActiveColor = ThemeColors.GetColorForTargetLuminance(ThemeColors.AverageColors(mThemeColors.DistinctColorB, mThemeColors.MiddleColor), 0.76, false);
                mMineEmergeColor = ThemeColors.GetColorForTargetLuminance(ThemeColors.AverageColors(mThemeColors.DistinctColorA, mThemeColors.DistinctColorB), 0.68, false);
                mMineBackgroundColor = ThemeColors.AdjustColorsForMultipler(0.333, mThemeColors.AveragedColor);
            }
            mMineXColorDark = ThemeColors.AdjustColorsForMultipler(0.75, mMineDormantColor);
            mMineXColorBright = ThemeColors.AdjustColorsForMultipler(1.25, mMineDormantColor);
            mMineActiveXColorDark = ThemeColors.AdjustColorsForMultipler(1.1, mMineActiveColor);
            mMineActiveXColorBright = ThemeColors.AdjustColorsForMultipler(1.2, mMineActiveColor);
        }

        public void EnableDisableMines(bool disable) { mDisableMines = disable; }

        public int GetCurrentMineCount()
        {
            return mMineCenters.Count;
        }

        public void InitializeMines(DrawRect bounds, double drawArea, double mineSizeMult = 1, bool minesMoreCentered = false, bool smallMines = false)
        {
            if (mDisableMines) { return; }

            mSkipRemoveActive = false;

            mVoidStormMineSizeRandomizers.Clear();
            mMineRadiusMult = 1;

            mMineCenters.Clear();
            mMineRadiuses.Clear();

            mMineBaseSize = drawArea * MinBaseUpdateDistance;

            MinDistFromFingerToSpawn = (float)(MinDistFromFingerToSpawnBase * mMineBaseSize);
            MinDistFromFingerMax = (float)(MinDistFromFingerToSpawnMaxBase * mMineBaseSize);

            // TODO: Make it so this class doesn't need to know about game modes
            if (mPlayMode == PlayMode.VoidStormV2)
            {
                mMineRadiusMiddle = (float)(mMineBaseSize * MineVoidStormRadiusBase);
            }
            else
            {
                if (smallMines)
                {
                    mMineRadiusMiddle = (float)(mMineBaseSize * MineSmallRadiusBase);
                }
                else
                {
                    mMineRadiusMiddle = (float)(mMineBaseSize * MineRadiusBase);
                }
            }
            mMineRadiusMiddle *= mineSizeMult;

            float margin = 2f;
            MineMinX = (float)(bounds.MinX + mMineRadiusMiddle) + margin;
            MineMaxX = (float)(bounds.MaxX - mMineRadiusMiddle) - margin;
            MineMinY = (float)(bounds.MinY + mMineRadiusMiddle) + margin;
            MineMaxY = (float)(bounds.MaxY - mMineRadiusMiddle) - margin;

            if (minesMoreCentered)
            {
                // Make the circles get placed more centrally on screen.
                MineMinX += (float)mMineBaseSize;
                MineMaxX -= (float)mMineBaseSize;
                MineMinY += (float)mMineBaseSize;
                MineMaxY -= (float)mMineBaseSize;
            }
        }

        // TODO: Re-factor this to be partially private (the init code), then a special function to pulsate, and to draw without initialization
        // TODO: This should no longer be needed, since we never start with mines. But it does other needed init stuff?
        public void DrawStartMines(bool init)
        {
            if (mDisableMines) { return; }

            const float baseColorMult = SharedParams.TwoThirds;
            DrawColor color = mThemeColors.Color3;
            if (init)
            {
                color = ThemeColors.AdjustColorsForMultipler(baseColorMult, mThemeColors.Color3.R, mThemeColors.Color3.G, mThemeColors.Color3.B);
                mMines.Clear();

                mHitMines.Clear();
                mEmergeStates.Clear();

                for (int j = 0; j < mMineCenters.Count; j++)
                {
                    DrawNode Mine = new DrawNode();
                    mLayer.AddChild(Mine);
                    mMines.Add(Mine);
                    mHitMines.Add(false);
                    mEmergeStates.Add(RadiusEmergeSteps);
                }
            }
            else
            {
                foreach (DrawNode circle in mMines)
                {
                    circle.Clear();
                    circle.Cleanup();
                    circle.Visible = false;
                }
            }

            for (int j = 0; j < mMineCenters.Count; j++)
            {
                DrawSingleMine(j, color, (float)(mMineRadiuses[j]));
            }

            if (!init)
            {
                foreach (DrawNode circle in mMines)
                {
                    circle.Visible = true;
                }
            }
        }

        private void DrawSingleMine(int index, DrawColor color, float radius, float radiusPercent = 1, bool active = false)
        {
            radius *= radiusPercent;

            if (mEmergeStates[index] > 0 && mEmergeStates[index] < RadiusEmergeSteps)
            {
                // Draw 15% larger for emerging mines, in order to get a "bounce back" effect
                radius *= 1.125f * (RadiusEmergeSteps / (RadiusEmergeSteps - 1));
            }

            mMines[index].Clear();
            mMines[index].Cleanup();
            mMines[index].Visible = false;

            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - radius, mMineCenters[index].Y),
                to: new DrawPoint(mMineCenters[index].X + radius, mMineCenters[index].Y),
                color: mMineBackgroundColor,
                lineWidth: radius,
                lineCap: CCLineCap.Butt);

            float width = radius * 0.111f;
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - radius, mMineCenters[index].Y+radius),
                to: new DrawPoint(mMineCenters[index].X + radius, mMineCenters[index].Y+radius),
                color: color,
                lineWidth: width,
                lineCap: CCLineCap.Round);
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - radius, mMineCenters[index].Y - radius),
                to: new DrawPoint(mMineCenters[index].X + radius, mMineCenters[index].Y - radius),
                color: color,
                lineWidth: width,
                lineCap: CCLineCap.Round);
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - radius, mMineCenters[index].Y - radius),
                to: new DrawPoint(mMineCenters[index].X - radius, mMineCenters[index].Y + radius),
                color: color,
                lineWidth: width,
                lineCap: CCLineCap.Round);
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X + radius, mMineCenters[index].Y - radius),
                to: new DrawPoint(mMineCenters[index].X + radius, mMineCenters[index].Y + radius),
                color: color,
                lineWidth: width,
                lineCap: CCLineCap.Round);

            DrawMineX(index, radius, active);

            mMines[index].Visible = true;
        }

        void DrawMineX(int index, float radius, bool active)
        {
            DrawColor darkColor;
            DrawColor brightColor;
            if (active)
            {
                darkColor = mMineActiveXColorDark;
                brightColor = mMineActiveXColorBright;
            }
            else
            {
                darkColor = mMineXColorDark;
                brightColor = mMineXColorBright;
            }

            float mXLen = radius * 0.5f;
            float lineWidth1 = radius * 0.2f;
            if (active)
            {
                mXLen *= 2.125f;
                lineWidth1 *= 0.75f;
            }
                
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - mXLen, mMineCenters[index].Y - mXLen),
                to: new DrawPoint(mMineCenters[index].X + mXLen, mMineCenters[index].Y + mXLen),
                color: darkColor,
                lineWidth: lineWidth1,
                lineCap: CCLineCap.Round);
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X + mXLen, mMineCenters[index].Y - mXLen),
                to: new DrawPoint(mMineCenters[index].X - mXLen, mMineCenters[index].Y + mXLen),
                color: darkColor,
                lineWidth: lineWidth1,
                lineCap: CCLineCap.Round);

            // Draw a brightended X inside
            lineWidth1 *= 0.5f;
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X - mXLen, mMineCenters[index].Y - mXLen),
                to: new DrawPoint(mMineCenters[index].X + mXLen, mMineCenters[index].Y + mXLen),
                color: brightColor,
                lineWidth: lineWidth1,
                lineCap: CCLineCap.Round);
            mMines[index].DrawLine(
                from: new DrawPoint(mMineCenters[index].X + mXLen, mMineCenters[index].Y - mXLen),
                to: new DrawPoint(mMineCenters[index].X - mXLen, mMineCenters[index].Y + mXLen),
                color: brightColor,
                lineWidth: lineWidth1,
                lineCap: CCLineCap.Round);
        }

        // This is called from an Update function
        public void DrawEmergingAndDecayingMines()
        {
            for (int j = 0; j < mEmergeStates.Count; j++)
            {
                if (mEmergeStates[j] > 0 && mEmergeStates[j] <= RadiusEmergeSteps)
                {
                    DrawSingleMine(j, mEmergeStates[j] < RadiusEmergeSteps ? mMineEmergeColor : mMineDormantColor, (float)mMineRadiuses[j], (float)mEmergeStates[j] / (float)RadiusEmergeSteps);

                    mEmergeStates[j]++;
                }
                else if (mEmergeStates[j] < 0)
                {
                    DrawSingleMine(j, mMineDormantColor, (float)mMineRadiuses[j], (float)-mEmergeStates[j] / (float)RadiusEmergeSteps);

                    mEmergeStates[j]++;
                }
                else if (mEmergeStates[j] == 0)
                {
                    RemoveMine(j);
                }
            }
        }

        public void SetMineRadiusMult(double mult)
        {
            mMineRadiusMult = mult;
        }

        public void AddRandomMine(DrawPoint finger)
        {
            if (mDisableMines) { return; }

            Random rnd = new Random();

            DrawPoint center = new DrawPoint();
            bool overlap = true;

            float minDist = (float)(mMineRadiusMiddle * 15) / (float)Math.Sqrt(mMineCenters.Count + 1);
            if (minDist < mMineRadiusMiddle * 3) { minDist = (float)(mMineRadiusMiddle * 3); }

            float minDistFromFinger = MinDistFromFingerToSpawn;
            if (minDistFromFinger < mMineRadiusMiddle * MinMineRadiusesToSpawn) { minDistFromFinger = (float)mMineRadiusMiddle * MinMineRadiusesToSpawn; }    
            if (minDistFromFinger > MinDistFromFingerMax) { minDistFromFinger = MinDistFromFingerMax; }

            float maxDistFromFinger = minDistFromFinger * 2;

            int iter = 0;
            while (overlap && iter < 2000)
            {
                iter++;

                center.X = (float)((rnd.NextDouble() * (MineMaxX - MineMinX)) + MineMinX);
                center.Y = (float)((rnd.NextDouble() * (MineMaxY - MineMinY)) + MineMinY);

                bool ok = true;
                if ((Math.Abs(finger.X - center.X) <= minDistFromFinger &&
                    Math.Abs(finger.Y - center.Y) <= minDistFromFinger) ||
                    (Math.Abs(finger.X - center.X) > maxDistFromFinger &&
                    Math.Abs(finger.Y - center.Y) > maxDistFromFinger))
                {
                    ok = false;
                }

                if (ok)
                {
                    foreach (DrawPoint pt in mMineCenters)
                    {
                        if ((Math.Abs(pt.X - center.X) <= minDist &&
                            Math.Abs(pt.Y - center.Y) <= minDist))
                        {
                            ok = false;
                            break;
                        }
                    }
                }

                // When we can't place it, reduce the distance from other circles requirement, but not the min distance from finger requirement.
                minDist *= 0.995f;
                maxDistFromFinger *= 0.995f;

                if (ok) { overlap = false; }
            }

            mMineCenters.Add(center);
            double radius = mMineRadiusMiddle * mMineRadiusMult;
            if (mPlayMode == PlayMode.VoidStormV2)
            {
                radius *= GetMineRadiusRandomizer(ref rnd);
            }
            mMineRadiuses.Add(radius);

            DrawNode Mine = new DrawNode();
            mLayer.AddChild(Mine);
            mMines.Add(Mine);
            mHitMines.Add(false);
            mEmergeStates.Add(1);
        }

        private double GetMineRadiusRandomizer(ref Random rnd)
        {
            // We do this instead, so we don't randomly get too many large or small mines
            double sizeMult = 1;
            if (mVoidStormMineSizeRandomizers.Count == 0)
            {
                try
                {
                    for (double d = (1 - VoidStormMineSizeRandomizer); d <= (1 + VoidStormMineSizeRandomizer + 0.001); d += IncVoidStormMineSizePerRandomizer)
                    {
                        mVoidStormMineSizeRandomizers.Add(d);
                    }
                }
                catch { }
            }

            if (mVoidStormMineSizeRandomizers.Count > 0)
            {
                try
                {
                    int index = rnd.Next(0, mVoidStormMineSizeRandomizers.Count);
                    sizeMult = mVoidStormMineSizeRandomizers[index];
                    mVoidStormMineSizeRandomizers.RemoveAt(index);
                }
                catch { }
            }

            return Math.Sqrt(sizeMult);
        }

        public void RemoveRandomMines(int count, bool skipLastAdded = false)
        {
            if (mDisableMines) { return; }

            Random rnd = new Random();

            if (count > mMineCenters.Count) { count = mMineCenters.Count - 1; }

            for (int j = 0; j < count; j++)
            {
                int index = rnd.Next(0, skipLastAdded ? mMineCenters.Count - 1 : mMineCenters.Count);

                // Make sure it's a Normal index.
                // Max iters is because it's possible that there are no Normal mines
                int maxIters = 1000;
                int iter = 0;
                while (!(mEmergeStates[index] > RadiusEmergeSteps) && iter < maxIters)
                {
                    index = rnd.Next(0, mMineCenters.Count);
                    iter++;
                }

                if (mSkipRemoveActive && mHitMines[index])
                {
                    mSkipRemoveActive = false;
                    j--;
                }
                else if (iter != maxIters)
                {
                    mEmergeStates[index] = -(RadiusEmergeSteps - 1);
                }
            }

            mSkipRemoveActive = false;
        }

        void RemoveMine(int index)
        {
            mLayer.RemoveChild(mMines[index]);
            mMineCenters.RemoveAt(index);
            mMineRadiuses.RemoveAt(index);
            mMines.RemoveAt(index);
            mHitMines.RemoveAt(index);
            mEmergeStates.RemoveAt(index);
        }

        public bool HitTestMine(DrawPoint pt)
        {
            bool endRoundEarly = false;

            if (mDisableMines) { return endRoundEarly; }

            // Scale the MineTicksBetweeMineHitAndDeath based on the number of mines.
            // Note it's ok if this ends up negative, which happens when there are 100+ mines.
            long deathTicksDelay = (long)(MineTicksBetweeMineHitAndDeath - (Math.Sqrt(mMineCenters.Count) * ((mMineRadiusMiddle * mMineRadiusMult) / mMineBaseSize) * MineTicksDeathThresholdReductionPerSqrtMineCountXRadius));
            if (deathTicksDelay < MineTicksDeathThresholdReductionPerSqrtMineCountXRadius) { deathTicksDelay = MineTicksDeathThresholdReductionPerSqrtMineCountXRadius; }

            bool hit = false;
            for (int j = 0; j < mMineCenters.Count; j++)
            {
                float widthHeight = (float)mMineRadiuses[j] * 2f;
                DrawRect hitTestRect = new DrawRect(mMineCenters[j].X - (float)mMineRadiuses[j], mMineCenters[j].Y - (float)mMineRadiuses[j], widthHeight, widthHeight);
                if (hitTestRect.ContainsPoint(pt) && 
                    mEmergeStates[j] > RadiusEmergeSteps)
                {
                    // Hit
                    hit = true;
                    mSkipRemoveActive = true;

                    if (mHitMines[j] == true && mLastHitTestResult == false &&
                        DateTime.UtcNow.Ticks > mLastMineHitTicks + deathTicksDelay)
                    {
                        if (mPlayMode == PlayMode.VoidStormV2)
                        {
                            // Set for deletion.
                            MainActivity.SoundEffects.OrbCapture();
                            mEmergeStates[j] = -(RadiusEmergeSteps - 1); 
                        }
                        else
                        {
                            MainActivity.SoundEffects.RoundEndEarly();
                        }
                        DrawSingleMine(j, mMineActiveColor, (float)mMineRadiuses[j] * 3f, active: true);
                        endRoundEarly = true;
                    }
                    else
                    {
                        if (!mHitMines[j])
                        {
                            MainActivity.SoundEffects.HitDormantMine();
                            mLastMineHitTicks = DateTime.UtcNow.Ticks;
                        }
                        DrawSingleMine(j, mMineActiveColor, (float)mMineRadiuses[j] * 1.0833f, active: true);
                        mHitMines[j] = true;
                    }

                    break;
                }
            }

            mLastHitTestResult = hit;
            return endRoundEarly;
        }

        public bool HitTestMinesBetweenPoints(DrawPoint pt1, DrawPoint pt2)
        {
            // We only need to increment points for the min mine diameter. 
            double minDiameter = mMineRadiusMiddle;
            foreach (double r in mMineRadiuses)
            {
                if (r < minDiameter) { minDiameter = r; }
            }
            minDiameter *= 2;

            // Get the number of points
            double distance = (Math.Sqrt((Math.Pow(pt2.X - pt1.X, 2) + Math.Pow(pt2.Y - pt1.Y, 2))));
            int numPoints = Convert.ToInt32(Math.Ceiling(distance / minDiameter));

            // Get the points themselves
            var diff_X = pt2.X - pt1.X;
            var diff_Y = pt2.Y - pt1.Y;

            var interval_X = diff_X / (numPoints + 1);
            var interval_Y = diff_Y / (numPoints + 1);

            List<DrawPoint> ptList = new List<DrawPoint>();
            for (int i = 1; i <= numPoints; i++)
            {
                ptList.Add(new DrawPoint(pt1.X + interval_X * i, pt1.Y + interval_Y * i));
            }

            // Pass the points to HitTestMine
            foreach (DrawPoint pt in ptList)
            {
                if (HitTestMine(pt))
                {
                    return true;
                }
            }

            // We may have hit one or more mines, but we didn't end the round
            return false;
        }

        public void HideMines(bool hide = true)
        {
            foreach (DrawNode mine in mMines)
            {
                mine.Visible = !hide;
            }
        }
    }
}