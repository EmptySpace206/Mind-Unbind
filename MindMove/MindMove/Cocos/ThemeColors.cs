using System;
using System.Collections.Generic;

using CocosSharp;

namespace MindMove.Cocos
{
    public class ThemeColors
    {
        public DrawColor Color1 { get; private set; }
        public DrawColor Color2 { get; private set; }
        public DrawColor Color3 { get; private set; }

        public DrawColor BackgroundColor { get; private set; }
        public DrawColor AveragedColor { get; private set; }
        public DrawColor AveragedColor12 { get; private set; }
        public DrawColor AveragedColor13 { get; private set; }
        public DrawColor AveragedColor23 { get; private set; }

        public DrawColor Text { get; private set; }
        public DrawColor TextColor1 { get; private set; }
        public DrawColor TextColor2 { get; private set; }
        public DrawColor TextColor3 { get; private set; }
        public DrawColor TextColor13 { get; private set; }

        public DrawColor Overlay { get; private set; }
        public DrawColor OverlayDark { get; private set; }

        DrawNode mBackground; // For drawing the background
        readonly bool mOverrideBackground = false;

        public DrawColor DistinctColorA { get; private set; }
        public DrawColor DistinctColorB { get; private set; }
        public DrawColor MiddleColor { get; private set; }

        public bool DualColorGradient { get; private set; } = false; // Used for determine if this theme should use a 3, or 2, color gradient

        readonly int mColorIndex;

        const double bkgColorDivFactor = 19;

        // Used for partial random color theme changes
        static ThemeColors LastRandomTheme = null;
        public static bool[] LockColors = { false, false, false };

        // Saved custom color themes
        struct ColorTrio
        {
            public DrawColor Color1;
            public DrawColor Color2;
            public DrawColor Color3;
        }
        static List<ColorTrio> CustomColorThemes = null;

        public ThemeColors(int colorIndex = 0, bool skipStaticSettingsChanges = false)
        {
            mColorIndex = colorIndex;

            if (colorIndex == 1)
            {
                // Summer sunset
                Color1 = new DrawColor(255, 166, 73);
                Color2 = new DrawColor(3, 201, 226);
                Color3 = new DrawColor(208, 69, 129);

                BackgroundColor = ThemeColors.AdjustColorsForMultipler(0.12, 191, 161, 148);
                mOverrideBackground = true;

                DualColorGradient = true;
            }
            else if (colorIndex == 2)
            {
                // Chillwave theme
                Color1 = new DrawColor(0, 237, 213);
                Color2 = new DrawColor(212, 64, 212);
                Color3 = new DrawColor(75, 0, 255);
            }
            else if (colorIndex == 3)
            {
                // Primary colors
                Color1 = new DrawColor(16, 255, 16);
                Color2 = new DrawColor(48, 48, 255);
                Color3 = new DrawColor(255, 16, 16);
            }
            else if (colorIndex == 4)
            {
                // Meditation theme
                Color1 = new DrawColor(123, 197, 223);
                Color2 = new DrawColor(3, 180, 182);
                Color3 = new DrawColor(212, 175, 55);

                BackgroundColor = new DrawColor(22, 24, 26);
                mOverrideBackground = true;

                DualColorGradient = true;
            }
            else if (colorIndex == 5)
            {
                // Bright birthday cake theme
                Color1 = new DrawColor(170, 232, 158);
                Color2 = new DrawColor(218, 32, 217);
                Color3 = new DrawColor(55, 182, 212);
            }
            else if (colorIndex == 6)
            {
                // Slate
                Color1 = new DrawColor(161, 169, 180);
                Color2 = new DrawColor(140, 140, 140);
                Color3 = new DrawColor(120, 136, 154);
            }
            else if (colorIndex == 7)
            {
                // Flowers
                Color1 = new DrawColor(221, 152, 210);
                Color2 = new DrawColor(232, 217, 70);
                Color3 = new DrawColor(118, 38, 127);

                DualColorGradient = true;
            }
            else if (colorIndex == 8)
            {
                // Mint coffee
                Color1 = new DrawColor(195, 230, 214);
                Color2 = new DrawColor(62, 180, 137);
                Color3 = new DrawColor(110, 78, 54);

                BackgroundColor = new DrawColor(17, 10, 10);
                mOverrideBackground = true;
            }
            else if (colorIndex == 9)
            {
                // Wintery theme
                Color1 = new DrawColor(182, 223, 246);
                Color2 = new DrawColor(113, 166, 209);
                Color3 = new DrawColor(19, 173, 235);

                DualColorGradient = true;
            }
            else if (colorIndex == 10)
            {
                // Autumn theme
                Color1 = new DrawColor(246, 183, 55);
                Color2 = new DrawColor(201, 67, 67);
                Color3 = new DrawColor(118, 180, 63);

                DualColorGradient = true;
            }
            else if (colorIndex == 11)
            {
                // Spring theme
                Color1 = new DrawColor(242, 230, 72);
                Color2 = new DrawColor(217, 137, 217);
                Color3 = new DrawColor(0, 224, 168);
            }
            else if (colorIndex == 12)
            {
                if (LastRandomTheme != null && skipStaticSettingsChanges)
                {
                    Color1 = LastRandomTheme.Color1;
                    Color2 = LastRandomTheme.Color2;
                    Color3 = LastRandomTheme.Color3;
                }
                else
                {
                    SetRandomColor();
                }
            }
            else if (CustomColorThemes != null && CustomColorThemes.Count > 0 &&
                colorIndex > BaseNumColorThemes &&
                colorIndex <= BaseNumColorThemes + CustomColorThemes.Count)
            {
                int index = colorIndex - (BaseNumColorThemes + 1);
                Color1 = new DrawColor(CustomColorThemes[index].Color1.R, CustomColorThemes[index].Color1.G, CustomColorThemes[index].Color1.B);
                Color2 = new DrawColor(CustomColorThemes[index].Color2.R, CustomColorThemes[index].Color2.G, CustomColorThemes[index].Color2.B);
                Color3 = new DrawColor(CustomColorThemes[index].Color3.R, CustomColorThemes[index].Color3.G, CustomColorThemes[index].Color3.B);
            }
            else
            {
                mColorIndex = 0;

                // Dahnya's theme
                Color1 = new DrawColor(0, 192, 192);
                Color2 = new DrawColor(253, 99, 70);
                Color3 = new DrawColor(95, 0, 191);

                BackgroundColor = ThemeColors.AdjustColorsForMultipler(0.12, 74, 119, 255);
                mOverrideBackground = true;
            }

            AveragedColor = new DrawColor(
                (byte)((Color1.R + Color2.R + Color3.R) / 3),
                (byte)((Color1.G + Color2.G + Color3.G) / 3),
                (byte)((Color1.B + Color2.B + Color3.B) / 3));

            if (!mOverrideBackground)
            {
                double avgLum = Luminance(AveragedColor, false);
                double bkgDiv = 3 * bkgColorDivFactor * avgLum;

                BackgroundColor = new DrawColor(
                    (byte)((Color1.R + Color2.R + Color3.R) / bkgDiv),
                    (byte)((Color1.G + Color2.G + Color3.G) / bkgDiv),
                    (byte)((Color1.B + Color2.B + Color3.B) / bkgDiv),
                    255);
            }

            // Get the most distinct colors
            double diff12 = Math.Abs(Color1.R - Color2.R) + Math.Abs(Color1.G - Color2.G) + Math.Abs(Color1.B - Color2.B);
            double diff13 = Math.Abs(Color1.R - Color3.R) + Math.Abs(Color1.G - Color3.G) + Math.Abs(Color1.A - Color3.B);
            double diff23 = Math.Abs(Color2.R - Color3.R) + Math.Abs(Color2.G - Color3.G) + Math.Abs(Color2.A - Color3.B);
            if (diff12 > diff13 && diff12 > diff23)
            {
                DistinctColorA = Color1;
                DistinctColorB = Color2;
                MiddleColor = Color3;
            }
            else if (diff13 > diff12 && diff13 > diff23)
            {
                DistinctColorA = Color1;
                DistinctColorB = Color3;
                MiddleColor = Color2;
            }
            else
            {
                DistinctColorA = Color2;
                DistinctColorB = Color3;
                MiddleColor = Color1;
            }

            TextColor1 = GetColorForTargetLuminance(Color1, 0.667);
            TextColor2 = GetColorForTargetLuminance(Color2, 0.75);
            TextColor3 = GetColorForTargetLuminance(Color3, 0.55);
            Text = GetColorForTargetLuminance(AveragedColor, 0.85);

            AveragedColor12 = new DrawColor(
                (byte)((Color1.R + Color2.R) / 2),
                (byte)((Color1.G + Color2.G) / 2),
                (byte)((Color1.B + Color2.B) / 2));

            AveragedColor13 = new DrawColor(
                (byte)((Color1.R + Color3.R) / 2),
                (byte)((Color1.G + Color3.G) / 2),
                (byte)((Color1.B + Color3.B) / 2));

            AveragedColor23 = new DrawColor(
                (byte)((Color2.R + Color3.R) / 2),
                (byte)((Color2.G + Color3.G) / 2),
                (byte)((Color2.B + Color3.B) / 2));

            TextColor13 = GetColorForTargetLuminance(AveragedColor13, 0.8167);

            Overlay = new DrawColor(0, 0, 0, 255 - 25);
            OverlayDark = new DrawColor(0, 0, 0, 255 - 15);


            if (IsRandomColorTheme())
            {
                LastRandomTheme = this;
            }
        }

        const int BaseNumColorThemes = 12;
        public static int GetNumColorThemes()
        {
            int count = BaseNumColorThemes;
            if (CustomColorThemes != null)
            {
                count += CustomColorThemes.Count;
            }
            return count;
        }

        public static int GetNumCustomThemes()
        {
            if (CustomColorThemes != null)
            {
                return CustomColorThemes.Count;
            }
            else
            {
                return 0;
            }
        }

        public static void SetSingleRandomColorToChange(int index)
        {
            if (index < 3 && index >= 0)
            {
                LockColors[0] = LockColors[1] = LockColors[2] = true;
                LockColors[index] = false;
            }
        }

        public static void ClearSingleRandomColorSettings()
        {
            LockColors[0] = LockColors[1] = LockColors[2] = false;
        }

        public int GetColorIndex()
        {
            return mColorIndex;
        }

        public bool IsRandomColorTheme()
        {
            if (mColorIndex == BaseNumColorThemes)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool IsRandomColorThemeIndex(int index)
        {
            return (index == BaseNumColorThemes ? true : false);
        }

        public bool IsCustomColorTheme()
        {
            if (mColorIndex > BaseNumColorThemes)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public bool IsNonBuiltInColorTheme()
        {
            if (mColorIndex >= BaseNumColorThemes)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        void SetRandomColor()
        {
            Random rnd = new Random();

            // Random color.  TODO: Make this create better random colors.
            const double minColLum = 0.333;
            const double maxColLum = 0.833;
            const double minTotalLum = 1.6;

            int maxIters = 4000;
            for (int j = 0; j < maxIters; j++)
            {
                // Luminance requirement
                if (LastRandomTheme != null && LockColors[0])
                {
                    Color1 = LastRandomTheme.Color1;
                }
                else
                {
                    Color1 = new DrawColor((byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256));
                }
                double lumCol1 = Luminance(Color1, false);
                if (lumCol1 < minColLum || lumCol1 > maxColLum)
                {
                    continue;
                }

                if (LastRandomTheme != null && LockColors[1])
                {
                    Color2 = LastRandomTheme.Color2;
                }
                else
                {
                    Color2 = new DrawColor((byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256));
                }
                double lumCol2 = Luminance(Color2, false);
                if (lumCol2 < minColLum || lumCol2 > maxColLum)
                {
                    continue;
                }

                if (LastRandomTheme != null && LockColors[2])
                {
                    Color3 = LastRandomTheme.Color3;
                }
                else
                {
                    Color3 = new DrawColor((byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256), (byte)rnd.Next(0, 256));
                }
                double lumCol3 = Luminance(Color3);
                if (lumCol3 < minColLum || lumCol3 > maxColLum)
                {
                    continue;
                }

                if (lumCol1 + lumCol2 + lumCol3 < minTotalLum)
                {
                    continue;
                }

                // Hue requirements
                const double minHueDistance = 1.0 / 10.0;
                double hueCol1 = Hue(Color1);
                double hueCol2 = Hue(Color2);
                double hueCol3 = Hue(Color3);
                if (Math.Abs(hueCol1 - hueCol2) < minHueDistance ||
                    Math.Abs(hueCol1 - hueCol3) < minHueDistance ||
                    Math.Abs(hueCol2 - hueCol3) < minHueDistance)
                {
                    continue;
                }

                break;
            }
        }

        public void CreateBackground(ref CCLayer layer)
        {
            mBackground = new DrawNode();
            mBackground.IsOpacityCascaded = false;
            mBackground.IsColorModifiedByOpacity = false;
            mBackground.Opacity = 1;
            layer.AddChild(mBackground);

            mBackground.DrawLine(
                from: new DrawPoint(0, 0),
                to: new DrawPoint(layer.ContentSize.Width, 0),
                color: BackgroundColor,
                lineWidth: layer.ContentSize.Height);
        }

        public static DrawColor InvertColor(DrawColor color)
        {
            return new DrawColor((byte)(255 - color.R), (byte)(255 - color.G), (byte)(255 - color.B));
        }

        public static DrawColor AverageColors(DrawColor color1, DrawColor color2)
        {
            return new DrawColor(
                (byte)((color1.R + color2.R) / 2),
                (byte)((color1.G + color2.G) / 2),
                (byte)((color1.B + color2.B) / 2));
        }

        public static DrawColor AverageColorsWeighted(DrawColor color1, double weightColor1, DrawColor color2)
        {
            // NOTE: weightColor1 must be >= 0 && <= 1
            double weightColor2 = 1.0 - weightColor1;

            return new DrawColor(
                (byte)((color1.R * weightColor1 + color2.R * weightColor2)),
                (byte)((color1.G * weightColor1 + color2.G * weightColor2)),
                (byte)((color1.B * weightColor1 + color2.B * weightColor2)));
        }

        public static DrawColor AdjustColorsForMultipler(double colorMultipler, DrawColor color)
        {
            return AdjustColorsForMultipler(colorMultipler, color.R, color.G, color.B);
        }

        public static DrawColor ReduceColorForMultiplierCheap(double colorMultipler, DrawColor color)
        {
            return new DrawColor((byte)(color.R * colorMultipler), (byte)(color.G * colorMultipler), (byte)(color.B * colorMultipler));
        }

        public static DrawColor AdjustColorsForMultipler(double colorMultiplier, byte orgR, byte orgG, byte orgB)
        {
            // Check for cases where multiply the color would bring it above 255; in those cases, instead
            // add evenly amongst the other 2 colors.
            int colAdd_r = 0;

            int colAdd_g = 0;
            int colAdd_b = 0;
            int remainder = (int)((int)orgR * colorMultiplier) - 255;
            if (remainder > 0)
            {
                colAdd_g += (remainder / 2);
                colAdd_b += (remainder / 2);

                if (orgG > orgB)
                {
                    remainder = (colAdd_g + orgG) - 255;
                    if (remainder > 0) { colAdd_b += remainder; }
                }
                else
                {
                    remainder = (colAdd_b + orgB) - 255;
                    if (remainder > 0) { colAdd_g += remainder; }
                }
            }
            remainder = (int)((int)orgG * colorMultiplier) - 255;
            if (remainder > 0)
            {
                colAdd_r += (remainder / 2);
                colAdd_b += (remainder / 2);

                if (orgR > orgB)
                {
                    remainder = (colAdd_r + orgR) - 255;
                    if (remainder > 0) { colAdd_b += remainder; }
                }
                else
                {
                    remainder = (colAdd_b + orgB) - 255;
                    if (remainder > 0) { colAdd_r += remainder; }
                }
            }
            remainder = (int)((int)orgB * colorMultiplier) - 255;
            if (remainder > 0)
            {
                colAdd_r += (remainder / 2);
                colAdd_g += (remainder / 2);

                if (orgR > orgG)
                {
                    remainder = (colAdd_r + orgR) - 255;
                    if (remainder > 0) { colAdd_g += remainder; }
                }
                else
                {
                    remainder = (colAdd_g + orgG) - 255;
                    if (remainder > 0) { colAdd_r += remainder; }
                }
            }

            int newColR = (int)((int)orgR * colorMultiplier) + colAdd_r;
            int newColG = (int)((int)orgG * colorMultiplier) + colAdd_g;
            int newColB = (int)((int)orgB * colorMultiplier) + colAdd_b;

            // Finally, make sure no colors are above 255.
            if (newColR > 255) { newColR = 255; }
            if (newColG > 255) { newColG = 255; }
            if (newColB > 255) { newColB = 255; }

            return new DrawColor(
                Convert.ToByte(newColR),
                Convert.ToByte(newColG),
                Convert.ToByte(newColB));
        }


        public static DrawColor GetColorForTargetLuminance(DrawColor color, double targetLuminance, bool isText = true)
        {
            DrawColor retColor = color;
            double luminance = Luminance(retColor, isText);

            const double inc = 0.02;
            double thresholdLuminance = targetLuminance - (inc / 2);
            int tries = (int)(1.0 / inc);

            // Only brighten the colors, don't darken them from their default
            if (luminance < thresholdLuminance)
            {
                for (int j = 1; j <= tries; j++)
                {
                    retColor = AdjustColorsForMultipler(1.0 + inc * j, color.R, color.G, color.B);
                    luminance = Luminance(retColor);

                    if (luminance >= thresholdLuminance)
                    {
                        break;
                    }
                }
            }
            else if (!isText && luminance > thresholdLuminance)
            {
                for (int j = 1; j <= tries; j++)
                {
                    retColor = AdjustColorsForMultipler(1.0 - inc * j, color.R, color.G, color.B);
                    luminance = Luminance(retColor);

                    if (luminance <= thresholdLuminance)
                    {
                        break;
                    }
                }
            }

            return retColor;
        }

        private static double Luminance(DrawColor color, bool isText = true)
        {
            if (isText)
            {
                return Math.Sqrt((0.47 * Math.Pow(color.R, 2)) + (0.51 * Math.Pow(color.G, 2)) + (0.02 * Math.Pow(color.B, 2))) / 255.0;
            }
            else
            {
                return Math.Sqrt((0.299 * Math.Pow(color.R, 2)) + (0.587 * Math.Pow(color.G, 2)) + (0.114 * Math.Pow(color.B, 2))) / 255.0;
            }
        }

        private static double Hue(DrawColor color)
        {
            try
            {
                if (color.R > color.G && color.R > color.B)
                {
                    return (color.G - color.B) / (color.R - Math.Min(color.G, color.B));
                }
                else if (color.G > color.R && color.G > color.B)
                {
                    return 2.0 + (color.B - color.R) / (color.G - Math.Min(color.R, color.B));
                }
                else
                {
                    return 4.0 + (color.R - color.G) / (color.B - Math.Min(color.R, color.G));
                }
            }
            catch 
            {
                return 0;
            }
        }

        const string CustomColorThemePath = "ColorThemes\\";
        const int MaxCustomColorThemes = 2;
        // Should only need to be called once during app execution.
        public static void LoadSavedColorThemes()
        {
            if (CustomColorThemes == null)
            {
                CustomColorThemes = new List<ColorTrio>();
            }
            CustomColorThemes.Clear();

            for (int j = 0; j < MaxCustomColorThemes; j++)
            {
                ColorTrio colors = new ColorTrio();
                // First check existence of this color theme
                int R1 = MainActivity.GetSavedIntWithDefaultValue(256, CustomColorThemePath + j + "\\R1.xml");
                if (R1 != 256)
                {
                    colors.Color1.R = (byte)R1;
                    colors.Color1.G = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\G1.xml");
                    colors.Color1.B = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\B1.xml");

                    colors.Color2.R = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\R2.xml");
                    colors.Color2.G = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\G2.xml");
                    colors.Color2.B = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\B2.xml");

                    colors.Color3.R = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\R3.xml");
                    colors.Color3.G = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\G3.xml");
                    colors.Color3.B = (byte)MainActivity.GetSavedIntWithDefaultValue(128, CustomColorThemePath + j + "\\B3.xml");

                    CustomColorThemes.Add(colors);
                }
            }
        }

        // Note: Should always be called after LoadSavedColorThemes has been previously called during this app instance
        public static void SaveColorTheme(ThemeColors colorTheme, int saveIndex)
        {
            if (CustomColorThemes == null || saveIndex > CustomColorThemes.Count || saveIndex >= MaxCustomColorThemes)
            {
                return;
            }

            ColorTrio colors = new ColorTrio();
            colors.Color1.R = colorTheme.Color1.R;
            colors.Color1.G = colorTheme.Color1.G;
            colors.Color1.B = colorTheme.Color1.B;
            colors.Color2.R = colorTheme.Color2.R;
            colors.Color2.G = colorTheme.Color2.G;
            colors.Color2.B = colorTheme.Color2.B;
            colors.Color3.R = colorTheme.Color3.R;
            colors.Color3.G = colorTheme.Color3.G;
            colors.Color3.B = colorTheme.Color3.B;

            MainActivity.SaveIntValue(colors.Color1.R, CustomColorThemePath + saveIndex + "\\R1.xml");
            MainActivity.SaveIntValue(colors.Color1.G, CustomColorThemePath + saveIndex + "\\G1.xml");
            MainActivity.SaveIntValue(colors.Color1.B, CustomColorThemePath + saveIndex + "\\B1.xml");
            MainActivity.SaveIntValue(colors.Color2.R, CustomColorThemePath + saveIndex + "\\R2.xml");
            MainActivity.SaveIntValue(colors.Color2.G, CustomColorThemePath + saveIndex + "\\G2.xml");
            MainActivity.SaveIntValue(colors.Color2.B, CustomColorThemePath + saveIndex + "\\B2.xml");
            MainActivity.SaveIntValue(colors.Color3.R, CustomColorThemePath + saveIndex + "\\R3.xml");
            MainActivity.SaveIntValue(colors.Color3.G, CustomColorThemePath + saveIndex + "\\G3.xml");
            MainActivity.SaveIntValue(colors.Color3.B, CustomColorThemePath + saveIndex + "\\B3.xml");

            if (CustomColorThemes.Count > saveIndex)
            {
                CustomColorThemes.RemoveAt(saveIndex);
            }
            CustomColorThemes.Insert(saveIndex, colors);
        }
    }
}