# راهنمای انتشار در GitHub (فارسی)

## ۱) قبل از انتشار
- فایل CSV را فقط در صورت مجاز بودن انتشار به پوشه `data/` اضافه کن؛ به‌صورت پیش‌فرض Git آن را نادیده می‌گیرد.
- فایل اصلی فارسی `Spotify_Song_Recommender_Professional.ipynb` را تغییر نده؛ این مخزن یک نسخه مستقل انگلیسی و قابل انتشار است.
- مدل‌ها را با `python train_model.py` بساز؛ پوشه `models/` به‌صورت پیش‌فرض در Git ثبت نمی‌شود.
- تست‌ها را اجرا کن: `pytest -q`.

## ۲) ساخت مخزن
1. در GitHub گزینه New repository را انتخاب کن.
2. نام پیشنهادی: `spotify-song-recommender`
3. توضیح پیشنهادی: `Content-based music recommendation using audio features, scikit-learn, Streamlit, and FastAPI.`
4. مخزن را Public انتخاب کن، مگر اینکه هنوز منبع/مجوز دیتاست یا فایل‌ها را بررسی نکرده‌ای.
5. چون فایل‌های پروژه را آماده کرده‌ای، از ایجاد README تکراری/فایل‌های اولیه غیرضروری خودداری کن.

## ۳) آپلود با GitHub Desktop
1. GitHub Desktop را باز کن و File → Add Local Repository را انتخاب کن.
2. پوشه استخراج‌شده `Spotify_Song_Recommender_Portfolio_Final` را انتخاب کن.
3. اگر گفت repository نیست، گزینه Create a repository را بزن.
4. Commit message: `Prepare professional music recommendation portfolio project`
5. Publish repository را بزن و مطمئن شو repository روی Public تنظیم شده است.

## ۴) معرفی مخزن
- About description: `Content-based music recommendation using standardized audio features and nearest neighbors.`
- Topics: `python`, `machine-learning`, `recommendation-system`, `scikit-learn`, `streamlit`, `fastapi`, `data-science`
- README را پس از انتشار در مرورگر بررسی کن تا تصاویر و مسیرهای نسبی درست نمایش داده شوند.
- هرگز API key، رمز، دیتاستی که مجوزش را نمی‌دانی یا فایل مدل حجیم را Push نکن.

## ۵) مواردی که نباید ادعا شوند
- پروژه رسمی Spotify نیست و از Spotify API برای پیشنهادها استفاده نمی‌کند.
- امتیاز فاصله یا شباهت، درصد دقت نیست.
- برنامه هنوز آنلاین Deploy نشده مگر اینکه واقعاً استقرارش داده باشی.
- پیشنهادها هنوز با برچسب انسانی یا آزمایش کاربران ارزیابی نشده‌اند.

## ۶) مجوز مخزن
تا زمانی که تصمیم نگرفته‌ای کد را با چه مجوزی منتشر کنی، فایل LICENSE اضافه نکن. مخزن عمومی بدون License به‌معنی اعطای مجوز استفاده مجدد از کد نیست. همچنین مجوز کد و مجوز دیتاست دو موضوع جدا هستند.
