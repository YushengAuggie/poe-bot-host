# YouTube Downloader Bot

A Poe bot that downloads YouTube videos from a provided URL and sends them back as attachments.

## Features

- Downloads YouTube videos from standard URLs, shorts, and youtu.be links
- Handles age-restricted content when possible
- Returns videos as downloadable attachments
- Rate limited to 300 points per message

## Bot Settings

The YouTube Downloader Bot has the following settings:

- **File Attachments**: Disabled for input (bot doesn't accept file uploads)
- **Rate Card**: 300 points per message
- **Maximum File Size**: 25MB for downloaded videos

## Usage Instructions

1. Simply paste a YouTube URL in the chat
2. The bot will download the video and send it back as an attachment
3. Users can then save the video to their device

## Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/shorts/VIDEO_ID`

## Deployment

The bot is deployed to Modal as a serverless function:

```bash
cd /Users/yding/workspace_quora/poe_bots_2
modal deploy bots/youtube_downloader_bot.py
```

## Setting the Rate Card

To set or update the rate card, use one of the following methods:

### Using the new update_bot_settings scripts:

```bash
# Set rate card to default 300 points and disable attachments
export YOUTUBEDOWNLOADERBOT_ACCESS_KEY=your_access_key_here
python scripts/update_youtube_bot_settings.py

# Or specify a custom rate
python scripts/update_youtube_bot_settings.py --rate-card 500

# For advanced settings, use the generic script
python scripts/update_bot_settings.py YouTubeDownloaderBot --rate-card 300 --no-attachments
```

### Using the sync_bot_settings.py script:

```bash
export YOUTUBEDOWNLOADERBOT_ACCESS_KEY=your_access_key_here
python sync_bot_settings.py --bot YouTubeDownloaderBot --rate-card 300
```

## Error Handling

The bot includes comprehensive error handling for:

1. **Invalid URLs** - The bot validates all URLs before attempting to download
2. **Download Failures** - Various error states are handled (geo-restrictions, age restrictions, etc.)
3. **File Size Limitations** - Videos over 25MB are rejected with a clear error message
4. **Setting Configurations** - The get_settings method includes robust error handling

## Limitations

- Videos must be under 25MB to be delivered as attachments
- Cannot download age-restricted content that requires login in some cases
- Cannot download private or removed videos
- Geo-restricted videos may not be available depending on the server location

## Troubleshooting

If users encounter issues:

1. Verify the YouTube URL is correct and the video is public
2. Check that the video isn't too large (over 25MB)
3. For age-restricted content, be aware that some videos may not be downloadable
4. For persistent issues, check the bot logs on Modal

### Rate Card Issues

If you encounter issues setting the rate card:

1. Make sure the access key environment variable is set correctly
2. Try the direct API method using the `scripts/update_bot_settings.py` script
3. Check for API errors in the script output with the `--verbose` flag
4. Verify the bot name is exactly "YouTubeDownloaderBot" (case sensitive)

## File Structure

- `bots/youtube_downloader_bot.py` - Main bot implementation
- `scripts/update_bot_settings.py` - Generic script for updating bot settings
- `scripts/update_youtube_bot_settings.py` - Specific script for YouTube Downloader Bot
- `scripts/set_bot_rate_card.py` - Alternative script for setting rate cards
- `scripts/set_youtube_bot_rate_card.py` - Alternative specific script for YouTube bot
