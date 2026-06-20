using System;
using System.Collections.Generic;
using Android;
using Android.App;
using Android.OS;
using Android.Views;
using AndroidX.Fragment.App;
using AndroidX.ViewPager.Widget;
using Plugin.StoreReview;
using AndroidX.AppCompat.Widget;
using MindMove.Cocos;
using Xamarin.Essentials;

namespace MindMove
{
    [Activity(
        Label = "Help"
    )]
    public class Help : Activity
    {
        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            SetContentView(Resource.Layout.Help);

            var giveFeedback = FindViewById<AppCompatButton>(Resource.Id.c_giveFeedback);
            if (giveFeedback != null)
            {
                giveFeedback.Click += GiveFeedback_Click;
            }
            var openSource = FindViewById<AppCompatButton>(Resource.Id.c_openSource);
            if (openSource != null)
            {
                openSource.Click += OpenSource_Click;
            }
        }

        private void OpenSource_Click(object sender, EventArgs e)
        {
            string url = "https://github.com/EmptySpace206/Mind-Unbind/blob/main/MovePredictionEngine.py";
            try
            {
                Browser.OpenAsync(url, BrowserLaunchMode.SystemPreferred).GetAwaiter().GetResult();
            }
            catch { }
        }

        private void GiveFeedback_Click(object sender, EventArgs e)
        {
            CrossStoreReview.Current.OpenStoreReviewPage("com.minimalistapps.MindMove");
        }
    }
}