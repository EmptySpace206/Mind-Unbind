
using System;
using System.Collections.Generic;

using CocosSharp;

namespace MindMove.Cocos
{
    public class ScoredDraw
    {
        public List<DrawPoint> pts { get; private set; }

        DrawNode line;
        CCLayer layer;

        public readonly DrawColor color;
        float width;

        public readonly double score;
        public readonly double relativeAngleAvg;
        public readonly double relativeAngleRaw;

        public readonly double colorMult;

        public readonly DrawColor secondaryColor;

        public ScoredDraw(ref CCLayer _layer, List<DrawPoint> path, DrawColor _color, double _colorMult, double _score, float lineWidth, double _relativeAngleAvg = 0, double _relativeAngleRaw = 0, DrawColor? _secondaryColor = null)
        {
            layer = _layer;

            pts = new List<DrawPoint>(path);

            color = _color;
            colorMult = _colorMult;
            width = lineWidth;
            score = _score;
            relativeAngleAvg = _relativeAngleAvg;
            relativeAngleRaw = _relativeAngleRaw;

            if (_secondaryColor != null)
            {
                secondaryColor = _secondaryColor.Value;
            }
        }

        ~ScoredDraw()
        {
            MakeInvisible(true);
            pts.Clear();
        }

        public void Draw(bool drawAngles = false, DrawColor? angleColor = null, float angleWidth = 1, bool visible = true)
        {
            if (line == null)
            {
                line = new DrawNode();
                layer.AddChild(line);
            }

            DrawColor actualColor = color;
            if (drawAngles)
            {
                actualColor = ThemeColors.AdjustColorsForMultipler(0.375, actualColor);
            }
            else
            {
                actualColor = ThemeColors.AdjustColorsForMultipler(colorMult, actualColor);
            }

            DrawColor shadowColor = ThemeColors.ReduceColorForMultiplierCheap(0.667, actualColor);

            for (int j = 0; j < pts.Count - 1; j++)
            {
                line.DrawLine(
                    from: pts[j],
                    to: pts[j + 1],
                    color: shadowColor,
                    lineWidth: width*1.5f,
                    lineCap: CCLineCap.Round);
            }
            for (int j = 0; j < pts.Count - 1; j++)
            {
                line.DrawLine(
                    from: pts[j],
                    to: pts[j + 1],
                    color: actualColor,
                    lineWidth: width,
                    lineCap: CCLineCap.Round);
            }

            if (drawAngles && pts.Count > 1)
            {
                DrawAngle(angleColor.Value, angleWidth, 60f);
            }

            line.Visible = visible;
        }

        void DrawAngle(DrawColor angleColor, float angleWidth, float distance, bool ignoreColorMult = false)
        {
            DrawPoint start = pts[0];
            float xAvg = 0;
            float yAvg = 0;
            int counter = 0;
            for (int j = 1; j < pts.Count; j++)
            {
                xAvg += pts[j].X;
                yAvg += pts[j].Y;
                counter++;
            }
            xAvg /= counter;
            yAvg /= counter;
            DrawPoint end = new DrawPoint(xAvg, yAvg);

            // Make the line a fixed length
            float xDiff = end.X - start.X;
            float yDiff = end.Y - start.Y;
            float angle = (float)(Math.Atan2(yDiff, xDiff));

            // Draw a line in width-reducing segments (to indicate direction)
            double segLen = distance / 3.0;
            DrawPoint endAngle1 = new DrawPoint((float)(start.X + Math.Cos(angle) * segLen), (float)(start.Y + Math.Sin(angle) * segLen));
            DrawPoint endAngle2 = new DrawPoint((float)(endAngle1.X + Math.Cos(angle) * segLen), (float)(endAngle1.Y + Math.Sin(angle) * segLen));
            DrawPoint endAngle3 = new DrawPoint((float)(endAngle2.X + Math.Cos(angle) * segLen), (float)(endAngle2.Y + Math.Sin(angle) * segLen));

            double colorMult = 1;
            if (!ignoreColorMult) { colorMult = (score * (3.0 / 7.0)) + 1; }
            DrawColor _color = ThemeColors.AdjustColorsForMultipler(colorMult, angleColor);

            line.DrawLine(
                from: start,
                to: endAngle1,
                color: _color,
                lineWidth: angleWidth * 1.5f,
                lineCap: CCLineCap.Round);
            line.DrawLine(
                from: endAngle1,
                to: endAngle2,
                color: _color,
                lineWidth: angleWidth,
                lineCap: CCLineCap.Round);
            line.DrawLine(
                from: endAngle2,
                to: endAngle3,
                color: _color,
                lineWidth: angleWidth * 0.5f,
                lineCap: CCLineCap.Round);
        }

        public static double GetColorMultForHighlightFactor(double factor)
        {
            return (0.25) + (1.5 * factor);
        }

        public void DrawWithHighlight(double highlightFactor, DrawColor HFColor, DrawColor baseColor, float baseWidth)
        {
            if (line == null)
            {
                line = new DrawNode();
                layer.AddChild(line);
            }

            double mult = GetColorMultForHighlightFactor(highlightFactor);
            float actualWidth = baseWidth * (float)mult;
            double colorMult = mult;
            DrawColor actualColor = ThemeColors.AdjustColorsForMultipler(colorMult, ThemeColors.AverageColorsWeighted(HFColor, highlightFactor, baseColor));

            for (int j = 0; j < pts.Count - 1; j++)
            {
                line.DrawLine(
                    from: pts[j],
                    to: pts[j + 1],
                    color: actualColor,
                    lineWidth: actualWidth,
                    lineCap: CCLineCap.Round);
            }

            line.Visible = true;
        }

        public void ReAddToLayer()
        {
            layer.RemoveChild(line);
            layer.AddChild(line);
        }

        public void DrawStartEndIndicatorArrow(DrawColor color, float width, float distance)
        {
            DrawAngle(color, width, distance, true);
        }

        public void MakeInvisible(bool delete = false)
        {
            if (line != null)
            {
                try
                {
                    line.Clear();
                    line.Cleanup();
                    line.Visible = false;

                    if (delete)
                    {
                        line.Dispose();
                        layer.RemoveChild(line);
                        line = null;
                    }
                } catch { }
            }
        }
    }
}