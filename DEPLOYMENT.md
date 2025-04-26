# Deploying Poe Bot Host: A Step-by-Step Guide

This guide walks through the complete process of deploying your Poe bots to Modal and setting them up on the Poe platform. These instructions have been verified to work as of April 2025.

## Prerequisites

Before starting, make sure you have:

1. A [Modal](https://modal.com/) account
2. A [Poe](https://poe.com/) account
3. Your Poe Bot Host code ready to deploy
4. Python 3.7+ installed on your system

## Part 1: Deploying to Modal

### 1.1 Set Up Modal

1. **Install the Modal client** in your project environment:
   ```bash
   pip install modal-client
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

   This will open a browser window for authentication:

   ![Modal Auth](https://docs.modal.com/images/token_new.png)

### 1.2 Deploy Your Bots

1. **Deploy the entire framework** with all your bots:
   ```bash
   modal deploy app.py
   ```

   The output will look something like this:
   ```
   Deploying app.py...
   ✓ Created app 'poe-bot-host'. Visit it at https://modal.com/apps/poe-bot-host
   ✓ Created function 'fastapi_app'
   ✓ Created web endpoint 'https://yourusername--poe-bot-host-fastapi-app.modal.run'
   Deployment complete! Your Modal app is live at https://yourusername--poe-bot-host-fastapi-app.modal.run
   ```

   Make note of the URL - you'll need it to configure your bots on Poe.

### 1.3 Test Your Deployment

Verify that your deployment works by testing the following endpoints:

1. **Health check endpoint**:
   ```bash
   curl https://yourusername--poe-bot-host-fastapi-app.modal.run/health
   ```

   Expected response:
   ```json
   {
     "status": "ok",
     "version": "1.0.0",
     "bots": ["EchoBot", "ReverseBot", "UppercaseBot"],
     "bot_count": 3,
     "environment": {
       "debug": false,
       "log_level": "INFO",
       "allow_without_key": true
     }
   }
   ```

2. **Bot list endpoint**:
   ```bash
   curl https://yourusername--poe-bot-host-fastapi-app.modal.run/bots
   ```

   Expected response:
   ```json
   {
     "EchoBot": "A simple bot that echoes back the user's message.",
     "ReverseBot": "A bot that reverses the user's message.",
     "UppercaseBot": "A bot that converts the user's message to uppercase."
   }
   ```

## Part 2: Creating Bots on Poe

### 2.1 Access the Poe Creator Portal

1. Go to [creator.poe.com](https://creator.poe.com/)
2. Log in with your Poe account
3. Click "Create Bot" or "Create a bot" button

### 2.2 Configure Basic Bot Settings

Fill in the basic details for your bot:

1. **Name**: A unique name for your bot (e.g., "EchoBot")
2. **Description**: A clear description of what your bot does
3. **Profile picture**: Optional, upload an image for your bot
4. **Category**: Select a relevant category for your bot (e.g., "Tools" or "Entertainment")
5. **Default Initial Prompt**: Optional, provide context or instructions for your bot

### 2.3 Configure Server Settings

1. Scroll down to "Server Bot Settings" section and expand it
2. Select "Server bot" as the bot type
3. Configure the API:
   - **Server Endpoint**: Your Modal deployment URL + the specific bot path
     - Format: `https://yourusername--poe-bot-host-fastapi-app.modal.run/botname`
     - Example for EchoBot: `https://yourusername--poe-bot-host-fastapi-app.modal.run/echobot`
     - Note: Use lowercase for the bot name in the URL
     - Important: Make sure there are no trailing slashes at the end of the URL
   - **API Protocol**: Select "Poe Protocol" from the dropdown
   - **API Key Protection**: Select "Server Authorization"
   - **Authentication Token**: Enter `Bearer dummytoken`

### 2.4 Adding Bot Metadata (Optional but Recommended)

For a more polished bot experience, add:

1. **Sample Messages**: Add 3-5 example messages to help users know what to ask
   - Example for EchoBot: "Hello world", "Echo this message", "Repeat after me: test"
   - Example for WeatherBot: "Weather in New York", "What's the forecast for London?", "Tokyo weather"

2. **Knowledge Files**: If your bot requires reference materials, you can upload them here
   - Most bots in this framework don't need this, as they process input directly

3. **API Citation Preference**: Choose how your bot should cite sources
   - For most bots in this framework, "Don't cite sources" is appropriate

4. **Message feedback**: Choose whether to allow user feedback (recommended for gathering improvement ideas)

### 2.5 Save and Test

1. Click "Create Bot" to save your configuration
2. After creation, you'll be redirected to a chat with your bot
3. Test your bot by sending a message
4. The message will be sent to your Modal-hosted API, processed by your bot, and the response will be displayed in the chat

If your bot doesn't respond or you encounter errors, refer to the troubleshooting section below.

## Part 3: Troubleshooting

### 3.1 Modal Deployment Issues

If your Modal deployment fails:

1. **Check your code**: Ensure there are no syntax errors or import issues
2. **Review logs**: Use `modal app logs poe-bot-host` to see detailed error messages
3. **Verify requirements**: Make sure all required packages are in requirements.txt
4. **Check Python version**: Modal uses Python 3.10 by default

### 3.2 Bot Connection Issues

If your bot isn't responding correctly on Poe:

1. **Verify URL**: Double-check the server endpoint URL is correct
   - Ensure there are no trailing slashes at the end of the URL
   - Confirm that the bot name in the URL is lowercase
   - For example: `https://yourusername--poe-bot-host-fastapi-app.modal.run/echobot`

2. **Test API directly** using curl to see if your deployment works:
   ```bash
   curl -X POST "https://yourusername--poe-bot-host-fastapi-app.modal.run/echobot" \
     -H "Content-Type: application/json" \
     -d '{
       "version": "1.0",
       "type": "query",
       "query": [
           {"role": "user", "content": "Hello world!"}
       ],
       "user_id": "test_user",
       "conversation_id": "test_convo_123",
       "message_id": "test_msg_123",
       "protocol": "poe"
     }'
   ```

3. **Check Modal logs** to see if your server is receiving requests:
   ```bash
   modal app logs poe-bot-host
   ```

4. **Check API protocol**: Verify you selected "Poe Protocol" in the bot settings

5. **Verify deployment is active**: Check in the Modal dashboard that your app is running

### 3.3 Common Errors

1. **"Server not responding"**: Your Modal deployment might not be active or URL might be incorrect
2. **"Got an error from the server"**: Your bot code is likely throwing an exception
3. **"Authentication failed"**: If using API key protection, ensure the key is correct in both your code and Poe configuration

## Part 4: Updating Your Bots

### 4.1 Making Changes

1. Update your code locally
2. Test with `./run_local.sh`
3. Redeploy with `modal deploy app.py`
4. The existing deployment will be updated with your changes

### 4.2 Adding New Bots

1. Create a new bot file in the `bots` directory
2. Implement your bot by extending the `BaseBot` class
3. Deploy the updated framework with `modal deploy app.py`
4. Create a new bot configuration on Poe as described in Part 2

## Part 5: Monitoring and Maintenance

### 5.1 Modal Dashboard

1. Visit [modal.com/apps](https://modal.com/apps) to see your apps
2. Click on your `poe-bot-host` app to view detailed metrics
3. Monitor usage, errors, and performance

### 5.2 Poe Analytics

1. Go to the [Poe Creator Portal](https://creator.poe.com/)
2. Select your bot to view analytics
3. Monitor user interactions, ratings, and usage statistics

---

If you encounter any issues or have questions, please refer to:

- [Modal Documentation](https://modal.com/docs)
- [Poe Documentation](https://creator.poe.com/docs)
- [fastapi-poe Documentation](https://github.com/poe-platform/fastapi-poe)
