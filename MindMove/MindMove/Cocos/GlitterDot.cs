using System;
using CocosSharp;

namespace MindMove.Cocos
{
    public class GlitterDot
    {
        DrawNode node;

        DrawPoint pt;
        DrawPoint? travelPt;
        DrawColor color;
        float radius;

        long startDrawTicks;
        long endDrawTicks;
        long duration;
        long lastUpdate = 0;

        float radiusMult = 1;

        readonly double MinColorMult = 0.667;
        readonly double MaxColorMult = 1.333;

        bool layered;

        CCLayer layer;

        public GlitterDot(ref CCLayer _layer, DrawPoint _pt, DrawColor _color, float _radius, double minColorMult = 0, double maxColorMult = 0, bool _layered = false, DrawPoint? _travelPt = null)
        {
            pt = _pt;
            travelPt = _travelPt;
            color = _color;
            radius = _radius;

            if (minColorMult > 0 && maxColorMult > 0)
            {
                MinColorMult = minColorMult;
                MaxColorMult = maxColorMult;
            }

            layered = _layered;

            if (layered)
            {
                // We do this because we get extra brightness with layering
                MinColorMult *= 0.86f;
                MaxColorMult *= 0.86f;
                radius *= 1.12f;
            }

            layer = _layer;

            node = new DrawNode();
            node.IsOpacityCascaded = true;
            node.IsColorModifiedByOpacity = true;
        }

        ~GlitterDot()
        {
            Clear();
        }

        public void Clear()
        {
            if (node != null)
            {
                node.Clear();
                node.Dispose();
                layer.RemoveChild(node);
                node = null;
            }
        }

        public void SetStartEndDrawTicks(long _startDrawTicks, long _endDrawTicks)
        {
            startDrawTicks = _startDrawTicks;
            endDrawTicks = _endDrawTicks;
            duration = endDrawTicks - startDrawTicks;
        }

        public void SetRadiusMultiplier(double mult)
        {
            radiusMult = (float)mult;
        }

        public void ReAddToLayer()
        {
            if (node != null)
            {
                layer.AddChild(node);
                node.Visible = true;
            }
        }

        public void SetTerminateIfDrawNotStarted()
        {
            if (lastUpdate == 0)
            {
                endDrawTicks = startDrawTicks = lastUpdate = DateTime.UtcNow.Ticks;
            }
        }

        // Returns true when hidden after being shown, indicating this GlitterDot is no longer needed
        public bool ShowOrHide()
        {
            bool retval = false;

            if (node != null)
            {
                long now = DateTime.UtcNow.Ticks;

                if (now > endDrawTicks)
                {
                    Clear();
                    retval = true;
                }
                else if (now > startDrawTicks)
                {
                    if (lastUpdate == 0)
                    {
                        DrawColor minColor = ThemeColors.AdjustColorsForMultipler(MinColorMult, color.R, color.G, color.B);
                        node.DrawSolidCircle(pt, radius * radiusMult * (float)MinColorMult, minColor);
                        layer.AddChild(node);
                        lastUpdate = now;
                    }
                    else if (now - lastUpdate > SharedParams.PulsatePeriod * TimeSpan.TicksPerMillisecond)
                    {
                        // Brighten or darken the dot. Brighten it until half the duration, then darken it
                        double colorMult;
                        double percentDuration = (double)(now - startDrawTicks) / duration;
                        if (percentDuration < 0.5)
                        {
                            colorMult = ((percentDuration * 2) * (MaxColorMult - MinColorMult)) + MinColorMult;
                        }
                        else
                        {
                            colorMult = MaxColorMult - ((percentDuration - 0.5) * 2 * (MaxColorMult - MinColorMult));
                        }

                        DrawColor adjColor = ThemeColors.AdjustColorsForMultipler(colorMult, color.R, color.G, color.B);

                        node.Clear();
                        node.Cleanup();
                        node.Visible = false;

                        DrawPoint pt1 = pt;
                        if (travelPt != null)
                        {
                            float t = (float)percentDuration;
                            pt1 = new DrawPoint(
                                ((1 - t) * pt.X) + (t * travelPt.Value.X),
                                ((1 - t) * pt.Y) + (t * travelPt.Value.Y));
                        }

                        if (layered)
                        {
                            // We set an alpha and draw multiple dots to get a layered brightness effect.
                            adjColor.A = 170;
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.888f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.777f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.666f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.555f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.444f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.333f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.222f, adjColor);
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult * 0.111f, adjColor);
                        }
                        else
                        {
                            node.DrawSolidCircle(pt1, radius * radiusMult * (float)colorMult, adjColor);
                        }

                        node.Visible = true;
                        lastUpdate = now;
                    }
                }
            }
            else
            {
                // Shouldn't be needed, but for safty so the caller can respond to being finished
                retval = true;
            }

            return retval;
        }
    }
}