# Gemini Image Handling Fix

## Table of Contents
- [Summary](#summary)
- [The Problem](#the-problem)
- [The Fix](#the-fix)
- [Why Two Fixes?](#why-two-fixes)
- [Verification](#verification)
- [Technical Details](#technical-details)
- [Usage Notes](#usage-notes)

## Summary

This document outlines the issue and fix for image handling in the Gemini bot implementation for Poe.

## The Problem

When image attachments were sent through the Poe API (via fastapi-poe), the Gemini bot couldn't process them correctly. The specific issue was:

1. When receiving attachments through the API, the actual image content was only available in the attachment object's `__dict__["content"]`
2. It wasn't accessible as a direct attribute (`attachment.content`), which our code expected
3. This is because fastapi-poe uses Pydantic models, which handle attribute access differently

## The Fix

Two critical fixes were implemented in `bots/gemini.py`:

1. In `_extract_attachments` method (around line 200):
   ```python
   # Ensure content attribute is directly accessible for each attachment
   for attachment in attachments:
       if not hasattr(attachment, "content") and "content" in attachment.__dict__:
           # Add content attribute directly to the attachment object
           content = attachment.__dict__["content"]
           # Use setattr to make content accessible as attribute even for Pydantic models
           object.__setattr__(attachment, "content", content)
           logger.debug(f"Fixed attachment content accessibility: {attachment.name}")
   ```

2. In `_process_media_attachment` method (around line 272):
   ```python
   # FIXED: If attachment has content in __dict__ but not as attribute,
   # make it accessible as attribute (fix for Pydantic models in the fastapi-poe library)
   if (not hasattr(attachment, "content") or getattr(attachment, "content", None) is None) and \
      hasattr(attachment, "__dict__") and "content" in attachment.__dict__:
       logger.debug(f"Fixing content attribute accessibility")
       try:
           content_from_dict = attachment.__dict__["content"]
           # Use object.__setattr__ to bypass Pydantic validation
           object.__setattr__(attachment, "content", content_from_dict)
           logger.debug(f"Successfully fixed content attribute accessibility")
       except Exception as e:
           logger.warning(f"Failed to fix content attribute accessibility: {e}")
   ```

## Why Two Fixes?

We implemented the fix in both methods as a defensive programming approach:

1. The fix in `_extract_attachments` handles attachments at the initial extraction stage
2. The fix in `_process_media_attachment` provides a fallback if attachment processing happens without going through extraction first
3. This redundancy ensures the fix works regardless of how the code paths evolve

## Verification

The fix was verified with:

1. Direct test of both methods using synthetic attachments that mimic the issue
2. Confirmation that the fix doesn't break normal attachments with direct attribute access
3. Tests with proper API key configuration

## Technical Details

- The core issue is related to how Pydantic models handle attribute access vs `__dict__` access
- We use `object.__setattr__()` to bypass Pydantic's validator mechanism
- This approach doesn't interfere with normal attachment processing
- Added detailed logging to trace the attachment processing path

## Usage Notes

When using the Gemini bot:

1. Make sure the `GOOGLE_API_KEY` environment variable is properly set
2. For images, the `multimodal_model_name` should be correctly configured (it is set to "gemini-1.5-flash-latest" by default)
3. Ensure the server has restarted after applying this fix
