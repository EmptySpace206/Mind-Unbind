using AndroidX.ConstraintLayout.Helper.Widget;
using CocosSharp;

namespace MindMove.Cocos
{
    // ========== Wrapper Structures ==========

    /// <summary>
    /// Wrapper for CCPoint - facilitates future CocosSharp replacement
    /// </summary>
    public struct DrawPoint
    {
        public float X;
        public float Y;

        public DrawPoint(float x, float y)
        {
            X = x;
            Y = y;
        }

        public static implicit operator CCPoint(DrawPoint p) => new CCPoint(p.X, p.Y);
        public static implicit operator DrawPoint(CCPoint p) => new DrawPoint(p.X, p.Y);
    }

    /// <summary>
    /// Wrapper for CCSize - facilitates future CocosSharp replacement
    /// </summary>
    public struct DrawSize
    {
        public float Width;
        public float Height;

        public DrawSize(float width, float height)
        {
            Width = width;
            Height = height;
        }

        public static implicit operator CCSize(DrawSize s) => new CCSize(s.Width, s.Height);
        public static implicit operator DrawSize(CCSize s) => new DrawSize(s.Width, s.Height);
    }

    /// <summary>
    /// Wrapper for CCRect - facilitates future CocosSharp replacement
    /// </summary>
    public struct DrawRect
    {
        private CCRect _internal;

        public DrawRect(float x, float y, float width, float height)
        {
            _internal = new CCRect(x, y, width, height);
        }

        public float MinX => _internal.MinX;
        public float MaxX => _internal.MaxX;
        public float MinY => _internal.MinY;
        public float MaxY => _internal.MaxY;
        public float MidX => _internal.MidX;
        public float MidY => _internal.MidY;
        public DrawPoint Center => _internal.Center;
        public DrawSize Size => _internal.Size;
        public bool ContainsPoint(DrawPoint point) => _internal.ContainsPoint(point);
        public DrawRect BoundingRect => _internal;

        public static implicit operator CCRect(DrawRect r) => r._internal;
        public static implicit operator DrawRect(CCRect r) => new DrawRect(r.MinX, r.MinY, r.Size.Width, r.Size.Height);
    }

    /// <summary>
    /// Wrapper for CCColor4B - facilitates future CocosSharp replacement
    /// </summary>
    public struct DrawColor
    {
        public byte R;
        public byte G;
        public byte B;
        public byte A;

        public DrawColor(byte r, byte g, byte b, byte a = 255)
        {
            R = r;
            G = g;
            B = b;
            A = a;
        }

        public static implicit operator CCColor4B(DrawColor c) => new CCColor4B(c.R, c.G, c.B, c.A);
        public static implicit operator DrawColor(CCColor4B c) => new DrawColor(c.R, c.G, c.B, c.A);
        public static implicit operator CCColor3B(DrawColor c) => new CCColor3B(c.R, c.G, c.B);
        public static implicit operator DrawColor(CCColor3B c) => new DrawColor(c.R, c.G, c.B, 255);
    }

    // ========== DrawNode Wrapper ==========

    /// <summary>
    /// Wrapper for CCDrawNode - facilitates future CocosSharp replacement
    /// </summary>
    public class DrawNode
    {
        private CCDrawNode _node;

        public DrawNode()
        {
            _node = new CCDrawNode();
        }

        public bool Visible { get => _node.Visible; set => _node.Visible = value; }
        public bool IsOpacityCascaded { get => _node.IsOpacityCascaded; set => _node.IsOpacityCascaded = value; }
        public bool IsColorModifiedByOpacity { get => _node.IsColorModifiedByOpacity; set => _node.IsColorModifiedByOpacity = value; }
        public DrawPoint Position { get => _node.Position; set => _node.Position = value; }
        public int ZOrder { get => _node.ZOrder; set => _node.ZOrder = value; }
        public byte Opacity { get => _node.Opacity; set => _node.Opacity = value; }
        public CCNode Parent => _node.Parent;
        public DrawRect BoundingRect => _node.BoundingRect;

        public void DrawLine(DrawPoint from, DrawPoint to, DrawColor color, float lineWidth, CCLineCap lineCap = CCLineCap.Butt)
        {
            _node.DrawLine(from, to, lineWidth, color, lineCap);
        }

        public void DrawCircle(DrawPoint center, float radius, DrawColor color)
        {
            _node.DrawCircle(center, radius, color);
        }

        public void DrawSolidCircle(DrawPoint center, float radius, DrawColor color)
        {
            _node.DrawSolidCircle(center, radius, color);
        }

        public void Clear() => _node.Clear();
        public void Cleanup() => _node.Cleanup();
        public void Dispose() => _node.Dispose();

        public CCNode GetInternalNode() => _node;

        // For CCLayer.AddChild/RemoveChild compatibility
        public static implicit operator CCNode(DrawNode node) => node?._node;
    }

    // ========== DrawLabel Wrapper ==========

    /// <summary>
    /// Wrapper for CCLabel - facilitates future CocosSharp replacement
    /// </summary>
    public class DrawLabel
    {
        private CCLabel _label;
        public bool Visible { get => _label.Visible; set => _label.Visible = value; }

        public DrawLabel(string text, string fontName, float fontSize)
        {
            _label = new CCLabel(text, fontName, fontSize, CCLabelFormat.SystemFont);
        }

        public DrawLabel(string text, string fontName, float fontSize, CCLabelFormat format)
        {
            _label = new CCLabel(text, fontName, fontSize, format);
        }

        public DrawLabel(string text, string fontName, float fontSize, DrawSize dimensions)
        {
            _label = new CCLabel(text, fontName, fontSize, dimensions, CCLabelFormat.SystemFont);
        }

        public DrawLabel(string text, string fontName, float fontSize, DrawSize dimensions, CCLabelFormat format)
        {
            _label = new CCLabel(text, fontName, fontSize, dimensions, format);
        }

        public string Text { get => _label.Text; set => _label.Text = value; }
        public float PositionX { get => _label.PositionX; set => _label.PositionX = value; }
        public float PositionY { get => _label.PositionY; set => _label.PositionY = value; }
        public DrawPoint Position { get => _label.Position; set => _label.Position = value; }
        public byte Opacity { get => _label.Opacity; set => _label.Opacity = value; }
        public CCNode Parent => _label.Parent;

        public DrawColor Color
        {
            get => new DrawColor(_label.Color.R, _label.Color.G, _label.Color.B, 255);
            set => _label.Color = new CCColor3B(value.R, value.G, value.B);
        }

        public CCTextAlignment HorizontalAlignment
        {
            get => _label.HorizontalAlignment;
            set => _label.HorizontalAlignment = value;
        }

        public void Cleanup() => _label.Cleanup();
        public void Dispose() => _label.Dispose();
        
        // Add explicit method to get internal node for operations that need it
        public CCNode GetInternalNode() => _label;

        // For CCLayer.AddChild/RemoveChild compatibility
        public static implicit operator CCNode(DrawLabel label) => label?._label;
    }

    // ========== DrawLayer Wrapper ==========

    /// <summary>
    /// Wrapper for CCLayer - facilitates future CocosSharp replacement
    /// </summary>
    /*public class DrawLayer
    {
        private CCLayer _layer;

        public DrawLayer()
        {
            _layer = new CCLayer();
        }

        public bool Visible { get => _layer.Visible; set => _layer.Visible = value; }
        public DrawPoint Position { get => _layer.Position; set => _layer.Position = value; }
        public float PositionX { get => _layer.PositionX; set => _layer.PositionX = value; }
        public float PositionY { get => _layer.PositionY; set => _layer.PositionY = value; }
        public int ZOrder { get => _layer.ZOrder; set => _layer.ZOrder = value; }
        public byte Opacity { get => _layer.Opacity; set => _layer.Opacity = value; }
        public CCNode Parent => _layer.Parent;
        public DrawSize ContentSize { get => _layer.ContentSize; set => _layer.ContentSize = value; }

        public DrawColor Color
        {
            get => new DrawColor(_layer.Color.R, _layer.Color.G, _layer.Color.B, 255);
            set => _layer.Color = new CCColor3B(value.R, value.G, value.B);
        }

        public void AddChild(CCNode child) => _layer.AddChild(child);
        public void AddChild(CCNode child, int zOrder) => _layer.AddChild(child, zOrder);
        public void RemoveChild(CCNode child) => _layer.RemoveChild(child);
        public void RemoveAllChildren() => _layer.RemoveAllChildren();

        public void Cleanup() => _layer.Cleanup();
        public void Dispose() => _layer.Dispose();

        public CCLayer GetInternalLayer() => _layer;

        // For CCScene.AddChild compatibility
        public static implicit operator CCLayer(DrawLayer layer) => layer._layer;
        public static implicit operator CCNode(DrawLayer layer) => layer._layer;
    }*/
}