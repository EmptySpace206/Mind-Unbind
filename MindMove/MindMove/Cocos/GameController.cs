using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using CocosSharp;

namespace MindMove.Cocos
{
    class GameController
    {
        public static CCGameView GameView
        {
            get;
            private set;
        }

        public static void Initialize(CCGameView gameView)
        {
            GameView = gameView;

            TitleScene scene = new TitleScene(GameView);
            GameView.RunWithScene(scene);
        }

        public static void PushScene(CCScene scene)
        {
            var transition = new CCTransitionCrossFade(0.11f, scene);
            GameView.Director.PushScene(transition);
        }

        public static void PopScene()
        {
            GameView.Director.PopScene();
        }
    }
}