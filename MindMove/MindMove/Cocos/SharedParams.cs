using Android.Hardware.Camera2;
using CocosSharp;
using System;
using System.Security.Cryptography;
using static MindMove.Cocos.GameScene;

namespace MindMove.Cocos
{
    public static class SharedParams
    {
        //
        // Frequently used values
        //
        public const float OneTwelvth = 1.0f / 12.0f;
        public const float OneNineth = 1.0f / 9.0f;
        public const float OneEigth = 1.0f / 8.0f;
        public const float OneSixth = 1.0f / 6.0f;
        public const float OneFourth = 1.0f / 4.0f;
        public const float OneThird = 1.0f / 3.0f;
        public const float TwoThirds = 2.0f / 3.0f;
        public const float ThreeFourths = 3.0f / 4.0f;
        public const float FourThirds = 4.0f / 3.0f;
        public const float FiveSixth = 5.0f / 6.0f;

        //
        // Pulsate parameters
        //
        public const int PulsatePeriod = 10;

        // For sizing based on screen dimensions
        public const float SizeAdjustRatioThreshold = 2.0f;

        // Standard update distance as percent of sqrt(draw area)
        public const double DefaultDistancePerUpdate = 1.0 / 10.5;
        public const double MinDistancePerUpdate = 1.0 / 11.667;
        public const double MaxDistancePerUpdate = 1.0 / 9.333;
        public const double UpdateDistanceSnapGrid = 1.0 / 11.0;
        public const double MaxUpdateDistanceDelta = 0.05;
        public const double UpdateDistanceAngleDeltaFactor = (1.0 / 6.0);

        //
        // For rank titles
        //
        public const string RankVoid_0 = "Seeker of Storms";
        public const string RankVoid_1 = "Storm Braver";
        public const string RankVoid_2 = "Void Survivor";
        public const string RankVoid_3 = "Chaos Champ";
        public const string RankVoid_4 = "Subduer of Storms";
        public const string RankVoid_5 = "Silencer of the Abyss";
        public const string RankBasic_0 = "Seeker of Randomness";
        public const string RankBasic_1 = "Conscious Changer";
        public const string RankBasic_2 = "Randomizer";
        public const string RankBasic_3 = "Skilled Scribbler";
        public const string RankBasic_4 = "Dynamic Doodler";
        public const string RankBasic_5 = "Continual Changer";

        public const double RankVoid_0Score = 0;
        public const double RankVoid_1Score = 90;
        public const double RankVoid_2Score = 100;
        public const double RankVoid_3Score = 110;
        public const double RankVoid_4Score = 120;
        public const double RankVoid_5Score = 130;

        public const double RankBasic_0Score = 0;
        public const double RankBasic_1Score = 92;
        public const double RankBasic_2Score = 100;
        public const double RankBasic_3Score = 108;
        public const double RankBasic_4Score = 112;
        public const double RankBasic_5Score = 120;

        public const double RoBMUnlockBaseScore = RankBasic_3Score;
        public const double StormLevel1Score = RankVoid_2Score;
        public const double StormLevel2Score = RankVoid_3Score;
        public const double StormLevel3Score = RankVoid_4Score;
        public const double StormLevel4Score = RankVoid_5Score;

        public const double Continuity_1Score = 100;
        public const double Continuity_2Score = 104;
        public const double Continuity_3Score = 106;
        public const double Continuity_4Score = 110;

        public const double VoidStormUnlock = 100;
        public const double ContinuityUnlock = 100;
        public const double ContinuityLongSeriesUnlock = Continuity_2Score;

        public const string StormLevel0 = "Void Looms";
        public const string StormLevel1 = "Stormy Edges";
        public const string StormLevel2 = "Entangled Streams";
        public const string StormLevel3 = "Chaos Blooms";
        public const string StormLevel4 = "Limits of Infinity";

        public const string TargetAnglesRing = "Long Game";
        public const string LongContinuitySeries = "Long Series";
        public const string StreamOfContinuity = "Configurations";
        public const string VoidStorm = "Void Storm";

        public const string Continuity1 = "Patterns";
        public const string Continuity2 = "Shapes";
        public const string Continuity3 = "Angles";
        public const string Continuity4 = "Waves";

        public const string StandardBestScore = "BasicHighestScoreV1.xml";
        public const string BestTargetAnglesScore = "BestTargetAnglesScore.xml";
        public const string VoidStormBestScore = "VoidStormBestScoreV1.xml";
        public const string ContinuityBestScore = "ContinuityBestScore.xml";
        public const string ContinuityBestScoreLongSeries = "ContinuityBestScoreLongSeries.xml";

        public static bool IsUnlockedVoidStorm = false;
        public static bool IsUnlockedContinuity = false;
        public static int TotalTitleUnlocks = 0;
        public const int NumTitleUnlocksPerAward = 3;

        public static void UpdateUnlocks()
        {
            double mindStreamBest = MainActivity.GetSavedIntWithDefaultValue(0, StandardBestScore) / 100.0;
            double voidStormBest = MainActivity.GetSavedIntWithDefaultValue(0, VoidStormBestScore) / 100.0;
            double ringBest = MainActivity.GetSavedIntWithDefaultValue(0, BestTargetAnglesScore) / 100.0;
            double continuityBest = MainActivity.GetSavedIntWithDefaultValue(0, ContinuityBestScore) / 100.0;

            IsUnlockedVoidStorm = mindStreamBest >= VoidStormUnlock;
            IsUnlockedContinuity = mindStreamBest >= ContinuityUnlock;

            TotalTitleUnlocks = 0;
            if (mindStreamBest >= RankBasic_1Score) { TotalTitleUnlocks++; }
            if (mindStreamBest >= RankBasic_2Score) { TotalTitleUnlocks++; }
            if (mindStreamBest >= RankBasic_3Score) { TotalTitleUnlocks++; }
            if (mindStreamBest >= RankBasic_4Score) { TotalTitleUnlocks++; }

            if (voidStormBest >= RankVoid_2Score) { TotalTitleUnlocks++; }
            if (voidStormBest >= RankVoid_3Score) { TotalTitleUnlocks++; }
            if (voidStormBest >= RankVoid_4Score) { TotalTitleUnlocks++; }
            if (voidStormBest >= RankVoid_5Score) { TotalTitleUnlocks++; }

            if (ringBest >= RankBasic_5Score) { TotalTitleUnlocks++; }

            if (continuityBest >= Continuity_1Score) { TotalTitleUnlocks++; }
            if (continuityBest >= Continuity_2Score) { TotalTitleUnlocks++; }
            if (continuityBest >= Continuity_3Score) { TotalTitleUnlocks++; }
            if (continuityBest >= Continuity_4Score) { TotalTitleUnlocks++; }
        }

        public static int GetNumUnlockedColorThemes()
        {
            if (TotalTitleUnlocks >= 11)
            {
                return TotalTitleUnlocks + NumTitleUnlocksPerAward + 1;
            }
            else
            {
                return 3 * ((TotalTitleUnlocks / NumTitleUnlocksPerAward) + 1);
            }
        }

        public static double GetRelativeAngle(double angle1, double angle2)
        {
            double deg1, deg2;
            if (angle2 < angle1) { deg1 = angle2; deg2 = angle1; }
            else { deg1 = angle1; deg2 = angle2; }

            return Math.Min(deg2 - deg1, 360 + (deg1 - deg2));
        }

        public static DrawColor GetRandomIdleGlitterColor(ThemeColors themeColors, Random random)
        {
            double weight = random.NextDouble();
            int rnd = RandomNumberGenerator.GetInt32(0, 3);
            if (rnd == 0) { return ThemeColors.AverageColorsWeighted(themeColors.Color1, weight, themeColors.Color2); }
            else if (rnd == 1) { return ThemeColors.AverageColorsWeighted(themeColors.Color1, weight, themeColors.Color3); }
            else { return ThemeColors.AverageColorsWeighted(themeColors.Color2, weight, themeColors.Color3); }
        }

        public static DrawColor GetColorPatternsColor(double angle, ThemeColors themeColors)
        {
            // Make the line color a gradient based on relative angle. 
            if (themeColors.DualColorGradient)
            {
                return ThemeColors.AverageColorsWeighted(themeColors.DistinctColorB, angle, themeColors.DistinctColorA);
            }
            else
            {
                if (angle < 0.5)
                {
                    return ThemeColors.AverageColorsWeighted(themeColors.DistinctColorA, 1.0 - angle * 2, themeColors.MiddleColor);
                }
                else
                {
                    double temp = angle - 0.5;
                    return ThemeColors.AverageColorsWeighted(themeColors.MiddleColor, 1.0 - temp * 2, themeColors.DistinctColorB);
                }
            }
        }

        public static double VirtualizeState(double absState, ref double lastState, ref double lastState2, ref double lastStateVirt, ref double latestRelativeAngle, bool useVirtualization = true)
        {
            double state;
            if (useVirtualization && lastState != StateNotInit && lastState2 != StateNotInit)
            {
                double relativeAngle1 = SharedParams.GetRelativeAngle(absState, lastState);
                double relativeAngle2 = SharedParams.GetRelativeAngle(absState, lastState2);

                bool clockwise;
                double diff = Math.Abs(absState - lastState);
                if ((diff < 180 && absState > lastState) || (diff >= 180 && absState < lastState))
                {
                    clockwise = true;
                }
                else
                {
                    clockwise = false;
                }

                double adjustedRelAngle = 0;
                double weight1 = relativeAngle1 / 180.0;
                double weight2 = relativeAngle2 / 180.0;
                if (weight1 + weight2 > 0)
                {
                    // This is (relativeAngle1^2 + relativeAngle2^2) / (relativeAngle1 + relativeAngle2)
                    // This is the 'contraharmonic mean' of the last 2 relative angles. 
                    // It biases angles to be sharper, to correct for a natural bias against sharp angles.
                    adjustedRelAngle = ((relativeAngle1 * weight1) + (relativeAngle2 * weight2)) / (weight1 + weight2);
                }

                double newAngleState;
                if (clockwise)
                {
                    newAngleState = lastStateVirt + adjustedRelAngle;
                    if (newAngleState > BaseMoveWeights.CircularRange) { newAngleState -= BaseMoveWeights.CircularRange; }
                }
                else
                {
                    newAngleState = lastStateVirt - adjustedRelAngle;
                    if (newAngleState < 0) { newAngleState += BaseMoveWeights.CircularRange; }
                }

                //System.Diagnostics.Debug.WriteLine("Clockwise: " + clockwise + ", s0: " + angleState0 + ", s1: " + angleState1Abs + ", diff: " + (Math.Abs(angleState0 - angleState1Abs)));

                latestRelativeAngle = SharedParams.GetRelativeAngle(newAngleState, lastStateVirt);
                state = newAngleState;
            }
            else
            {
                if (lastState != StateNotInit)
                {
                    latestRelativeAngle = SharedParams.GetRelativeAngle(absState, lastState);
                }
                state = absState;
            }
            lastState2 = lastState;
            lastState = absState;
            lastStateVirt = state;

            return state;
        }
    }
}