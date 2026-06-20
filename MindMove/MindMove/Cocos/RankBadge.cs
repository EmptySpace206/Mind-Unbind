using Android.App;
using Android.Content;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using CocosSharp;

namespace MindMove.Cocos
{
    public class RankBadge
    {
        CCLayer mLayer;
        ThemeColors mThemeColors;

        DrawPoint mCenter;
        DrawPoint mOriginalCenter;
        float mRadius;

        int mBadgeLevel;

        bool mSolidBadge;

        DrawNode mBadgeCircleRing;
        DrawNode mBadgeCircleDots;
        DrawNode mBadgeCircleBackground;

        DrawColor mBackgroundColor;

        bool mActive = true;

        const float mBadgeColorWeight = SharedParams.OneThird;

        DrawColor mMainRingColor1;
        DrawColor mMainRingColor2;
        DrawColor mMainRingColor3;
        DrawColor mStaticColor1;
        DrawColor mStaticColor2;
        DrawColor mStaticColor3;
        const int RingColorSteps = 150; // Tends to be a bit faster than pulsate rate
        int[] mMainRingColorPct = { 0, 0, RingColorSteps };
        const int ColorWeightAdjustBadgeLevel = 5;

        bool mDarken = false;

        bool mDotsOnly = false;
        bool mExtraLargeDots = false;

        public RankBadge(ref CCLayer layer, float badgeRadius, DrawPoint badgeCenter, ThemeColors themeColors, int badgeLevel, bool solidBadge, bool draw = false, bool darken = false, bool dotsOnly = false, bool extraLargeDots = false)
        {
            mLayer = layer;
            mThemeColors = themeColors;
            mCenter = badgeCenter;
            mOriginalCenter = mCenter;

            mBadgeLevel = badgeLevel;
            if (mBadgeLevel > 4) { mBadgeLevel = 4; } // Set the max to 4 layers deep

            mRadius = badgeRadius;
            mSolidBadge = solidBadge;
            mExtraLargeDots = extraLargeDots;

            mBadgeCircleBackground = new DrawNode();
            mLayer.AddChild(mBadgeCircleBackground);

            mBadgeCircleRing = new DrawNode();
            mLayer.AddChild(mBadgeCircleRing);

            mBadgeCircleDots = new DrawNode();
            mLayer.AddChild(mBadgeCircleDots);

            mBackgroundColor = ThemeColors.AdjustColorsForMultipler(0.075, mThemeColors.AveragedColor.R, mThemeColors.AveragedColor.G, mThemeColors.AveragedColor.B);
            mBackgroundColor.A = 96;

            mBadgeCircleBackground.IsOpacityCascaded = false;

            mMainRingColor1 = mThemeColors.DistinctColorA;
            mMainRingColor2 = mThemeColors.MiddleColor;
            mMainRingColor3 = mThemeColors.DistinctColorB;

            mStaticColor1 = mMainRingColor1;
            mStaticColor2 = mMainRingColor2;
            mStaticColor3 = mMainRingColor3;

            mDarken = darken;

            mDotsOnly = dotsOnly;

            if (draw)
            {
                PulsateBadgeRing(1.0);
            }
        }

        void AdjustColorWeight()
        {
            // For now, just rotate colors for all badges. Note that rank 5 will be same as rank 4.
            //if (mBadgeLevel < ColorWeightAdjustBadgeLevel) { return; }

            if (mMainRingColorPct[2] > 0 && mMainRingColorPct[1] >= 0 && mMainRingColorPct[0] == 0)
            {
                mMainRingColorPct[2]--;
                mMainRingColorPct[1]++;
            }
            else if (mMainRingColorPct[2] == 0 && mMainRingColorPct[1] > 0 && mMainRingColorPct[0] >= 0)
            {
                mMainRingColorPct[1]--;
                mMainRingColorPct[0]++;
            }
            else if (mMainRingColorPct[2] >= 0 && mMainRingColorPct[1] == 0 && mMainRingColorPct[0] > 0)
            {
                mMainRingColorPct[0]--;
                mMainRingColorPct[2]++;
            }

            float mult1 = (float)mMainRingColorPct[0] / RingColorSteps;
            float mult2 = (float)mMainRingColorPct[1] / RingColorSteps;
            float mult3 = (float)mMainRingColorPct[2] / RingColorSteps;

            mMainRingColor1.R = (byte)(Math.Round(mStaticColor1.R * mult1 + mStaticColor2.R * mult2 + mStaticColor3.R * mult3));
            mMainRingColor1.G = (byte)(Math.Round(mStaticColor1.G * mult1 + mStaticColor2.G * mult2 + mStaticColor3.G * mult3));
            mMainRingColor1.B = (byte)(Math.Round(mStaticColor1.B * mult1 + mStaticColor2.B * mult2 + mStaticColor3.B * mult3));

            mMainRingColor2.R = (byte)(Math.Round(mStaticColor1.R * mult3 + mStaticColor2.R * mult1 + mStaticColor3.R * mult2));
            mMainRingColor2.G = (byte)(Math.Round(mStaticColor1.G * mult3 + mStaticColor2.G * mult1 + mStaticColor3.G * mult2));
            mMainRingColor2.B = (byte)(Math.Round(mStaticColor1.B * mult3 + mStaticColor2.B * mult1 + mStaticColor3.B * mult2));

            mMainRingColor3.R = (byte)(Math.Round(mStaticColor1.R * mult2 + mStaticColor2.R * mult3 + mStaticColor3.R * mult1));
            mMainRingColor3.G = (byte)(Math.Round(mStaticColor1.G * mult2 + mStaticColor2.G * mult3 + mStaticColor3.G * mult1));
            mMainRingColor3.B = (byte)(Math.Round(mStaticColor1.B * mult2 + mStaticColor2.B * mult3 + mStaticColor3.B * mult1));
        }

        void DrawSingleRing(DrawPoint center, float radius, float dotRadius, DrawColor color, DrawColor dotColor)
        {
            if (mDotsOnly)
            {
                mBadgeCircleDots.DrawSolidCircle(center, dotRadius, color);
                DrawColor backgroundColor;
                if (mExtraLargeDots)
                {
                    backgroundColor = ThemeColors.ReduceColorForMultiplierCheap(0.2, color);
                    mBadgeCircleBackground.DrawSolidCircle(center, dotRadius * 1.8f, backgroundColor);
                }
                backgroundColor = ThemeColors.ReduceColorForMultiplierCheap(0.4, color);
                mBadgeCircleBackground.DrawSolidCircle(center, dotRadius*1.4f, backgroundColor);
            }
            else
            { 
                if (mSolidBadge || mBadgeLevel == 0)
                {
                    mBadgeCircleRing.DrawSolidCircle(center, radius, color);
                }
                else
                {
                    mBadgeCircleRing.DrawCircle(center, radius, color);
                    mBadgeCircleRing.DrawCircle(center, radius + 1, color);
                    mBadgeCircleRing.DrawCircle(center, radius - 1, color);
                    mBadgeCircleRing.DrawCircle(center, radius + 2, color);
                    mBadgeCircleRing.DrawCircle(center, radius - 2, color);
                    mBadgeCircleBackground.DrawSolidCircle(center, radius, mBackgroundColor);
                }
                mBadgeCircleDots.DrawSolidCircle(center, dotRadius, dotColor);
            }
        }

        void DrawInnerRings(bool invert, DrawPoint center, float radius, float dotRadius, int innerLevels, 
            DrawColor color1, DrawColor color2, DrawColor color3,
            DrawColor color1I, DrawColor color2I, DrawColor color3I)
        {
            if (innerLevels < 0) { return; }

            float dotRadMult = SharedParams.TwoThirds;
            if (mDotsOnly)
            {
                dotRadMult = 1.167f;
            }

            float subRadius = radius * SharedParams.TwoThirds;
            float subDotRadius = dotRadius * dotRadMult;

            DrawPoint subCenter1;
            DrawPoint subCenter2;
            DrawPoint subCenter3;
            if (invert)
            {
                subCenter3 = new DrawPoint(center.X, center.Y - radius);
                subCenter2 = new DrawPoint(
                    (float)(center.X + (radius * ((float)Math.Cos(-Math.PI * (1 + SharedParams.OneSixth))))),
                    (float)(center.Y + (radius * ((float)Math.Sin(-Math.PI * (1 + SharedParams.OneSixth))))));
                subCenter1 = new DrawPoint(
                    (float)(center.X + (radius * ((float)Math.Cos(-Math.PI * (1 + SharedParams.FiveSixth))))),
                    (float)(center.Y + (radius * ((float)Math.Sin(-Math.PI * (1 + SharedParams.FiveSixth))))));
            }
            else
            {
                subCenter3 = new DrawPoint(center.X, center.Y + radius);
                subCenter2 = new DrawPoint(
                    (float)(center.X + (radius * ((float)Math.Cos(Math.PI * (1 + SharedParams.OneSixth))))),
                    (float)(center.Y + (radius * ((float)Math.Sin(Math.PI * (1 + SharedParams.OneSixth))))));
                subCenter1 = new DrawPoint(
                    (float)(center.X + (radius * ((float)Math.Cos(Math.PI * (1 + SharedParams.FiveSixth))))),
                    (float)(center.Y + (radius * ((float)Math.Sin(Math.PI * (1 + SharedParams.FiveSixth))))));
            }

            // Change the colors passed to DrawInnerRings based on the color of the outer ring (make each weigh in 33% the outer ring)
            float badgeColorAntiWeight = 1 - mBadgeColorWeight;
            DrawInnerRings(!invert, subCenter1, subRadius, subDotRadius, innerLevels - 1, 
                color1, 
                new DrawColor((byte)(color2.R * badgeColorAntiWeight + color1.R * mBadgeColorWeight), (byte)(color2.G * badgeColorAntiWeight + color1.G * mBadgeColorWeight), (byte)(color2.B * badgeColorAntiWeight + color1.B * mBadgeColorWeight)),
                new DrawColor((byte)(color3.R * badgeColorAntiWeight + color1.R * mBadgeColorWeight), (byte)(color3.G * badgeColorAntiWeight + color1.G * mBadgeColorWeight), (byte)(color3.B * badgeColorAntiWeight + color1.B * mBadgeColorWeight)),
                color1I, color1I, color1I);
            DrawInnerRings(!invert, subCenter2, subRadius, subDotRadius, innerLevels - 1,
                new DrawColor((byte)(color1.R * badgeColorAntiWeight + color2.R * mBadgeColorWeight), (byte)(color1.G * badgeColorAntiWeight + color2.G * mBadgeColorWeight), (byte)(color1.B * badgeColorAntiWeight + color2.B * mBadgeColorWeight)),
                color2,
                new DrawColor((byte)(color3.R * badgeColorAntiWeight + color2.R * mBadgeColorWeight), (byte)(color3.G * badgeColorAntiWeight + color2.G * mBadgeColorWeight), (byte)(color3.B * badgeColorAntiWeight + color2.B * mBadgeColorWeight)),
                color2I, color2I, color2I);
            DrawInnerRings(!invert, subCenter3, subRadius, subDotRadius, innerLevels - 1,
                new DrawColor((byte)(color1.R * badgeColorAntiWeight + color3.R * mBadgeColorWeight), (byte)(color1.G * badgeColorAntiWeight + color3.G * mBadgeColorWeight), (byte)(color1.B * badgeColorAntiWeight + color3.B * mBadgeColorWeight)),
                new DrawColor((byte)(color2.R * badgeColorAntiWeight + color3.R * mBadgeColorWeight), (byte)(color2.G * badgeColorAntiWeight + color3.G * mBadgeColorWeight), (byte)(color2.B * badgeColorAntiWeight + color3.B * mBadgeColorWeight)),
                color3,
                color3I, color3I, color3I);

            DrawSingleRing(subCenter1, subRadius, subDotRadius, color1, color1I);
            DrawSingleRing(subCenter2, subRadius, subDotRadius, color2, color2I);
            DrawSingleRing(subCenter3, subRadius, subDotRadius, color3, color3I);
        }

        public void PulsateBadgeRing(double scaleFactor, bool invert = false, bool rotateColors = true)
        {
            if (mActive && mBadgeCircleRing != null)
            {
                mBadgeCircleRing.Clear();
                mBadgeCircleRing.Cleanup();
                mBadgeCircleRing.Visible = false;

                mBadgeCircleDots.Clear();
                mBadgeCircleDots.Cleanup();
                mBadgeCircleDots.Visible = false;

                mBadgeCircleBackground.Clear();
                mBadgeCircleBackground.Cleanup();
                mBadgeCircleBackground.Visible = false;

                int badgeLevel = mBadgeLevel;
                if (rotateColors) { AdjustColorWeight(); }
                if (badgeLevel == ColorWeightAdjustBadgeLevel) { badgeLevel--; }

                float radiusMult = (float)(1 + ((scaleFactor - 1) * SharedParams.OneTwelvth));
                float radius = radiusMult * mRadius;

                // Reduce the radius of each circle, when there are more badge levels
                for (int j = 0; j < badgeLevel; j++) 
                {
                    radius *= SharedParams.FiveSixth;
                }

                float dotRadius;
                double scaleFactorDot = scaleFactor;
                if (mSolidBadge)
                {
                    dotRadius = radius * (SharedParams.OneNineth * 1.5f); 
                }
                else
                {
                    dotRadius = radius * SharedParams.OneNineth;

                    if (!mDotsOnly) { scaleFactorDot = 1 + ((scaleFactor - 1) * 1.5); }
                }

                if (mDotsOnly)
                {
                    dotRadius *= 0.5f;
                }
                if (mExtraLargeDots)
                {
                    dotRadius *= 1.25f;
                }

                DrawColor color123 = ThemeColors.AdjustColorsForMultipler(scaleFactorDot, mThemeColors.AveragedColor.R, mThemeColors.AveragedColor.G, mThemeColors.AveragedColor.B);

                DrawColor color1 = ThemeColors.AdjustColorsForMultipler(scaleFactor, mMainRingColor1.R, mMainRingColor1.G, mMainRingColor1.B);
                DrawColor color2 = ThemeColors.AdjustColorsForMultipler(scaleFactor, mMainRingColor2.R, mMainRingColor2.G, mMainRingColor2.B);
                DrawColor color3 = ThemeColors.AdjustColorsForMultipler(scaleFactor, mMainRingColor3.R, mMainRingColor3.G, mMainRingColor3.B);
                DrawColor color12 = ThemeColors.AdjustColorsForMultipler(scaleFactorDot, mThemeColors.AveragedColor12.R, mThemeColors.AveragedColor12.G, mThemeColors.AveragedColor12.B);
                DrawColor color13 = ThemeColors.AdjustColorsForMultipler(scaleFactorDot, mThemeColors.AveragedColor13.R, mThemeColors.AveragedColor13.G, mThemeColors.AveragedColor13.B);
                DrawColor color23 = ThemeColors.AdjustColorsForMultipler(scaleFactorDot, mThemeColors.AveragedColor23.R, mThemeColors.AveragedColor23.G, mThemeColors.AveragedColor23.B);

                if (mDarken)
                {
                    double darkenFactor = SharedParams.OneThird;
                    if (mSolidBadge) { darkenFactor = 0.3; }
                    color123 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color123.R, color123.G, color123.B);
                    color1 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color1.R, color1.G, color1.B);
                    color2 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color2.R, color2.G, color2.B);
                    color3 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color3.R, color3.G, color3.B);
                    color12 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color12.R, color12.G, color12.B);
                    color13 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color13.R, color13.G, color13.B);
                    color23 = ThemeColors.AdjustColorsForMultipler(darkenFactor, color23.R, color23.G, color23.B);
                }

                DrawSingleRing(mCenter, radius, dotRadius, color123, color123);
                if (invert)
                {
                    DrawInnerRings(true, mCenter, radius, dotRadius, badgeLevel - 1, color2, color1, color3, color13, color23, color12);
                }
                else
                {
                    DrawInnerRings(false, mCenter, radius, dotRadius, badgeLevel - 1, color1, color2, color3, color23, color13, color12);
                }

                mBadgeCircleRing.Visible = true;
                mBadgeCircleDots.Visible = true;
                mBadgeCircleBackground.Visible = true;
            }
        }

        public void Deactivate()
        {
            mActive = false;

            if (mBadgeCircleRing != null)
            {
                mBadgeCircleRing.Clear();
                mLayer.RemoveChild(mBadgeCircleRing);
                mBadgeCircleRing.Cleanup();
                mBadgeCircleRing = null;
            }

            if (mBadgeCircleDots != null)
            {
                mBadgeCircleDots.Clear();
                mLayer.RemoveChild(mBadgeCircleDots);
                mBadgeCircleDots.Cleanup();
                mBadgeCircleDots = null;
            }

            if (mBadgeCircleBackground != null)
            {
                mBadgeCircleBackground.Clear();
                mLayer.RemoveChild(mBadgeCircleBackground);
                mBadgeCircleBackground.Cleanup();
                mBadgeCircleBackground = null;
            }
        }

        public void SetTempCenter(DrawPoint pt)
        {
            mCenter = pt;
        }

        public void RestoreCenter()
        {
            mCenter = mOriginalCenter;
        }

        ~RankBadge()
        {
            Deactivate();
        }
    }
}