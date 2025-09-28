# TODO: Fix Bot Command Issues

## Webhook Setup
- [x] Make WEBHOOK_URL dynamic using RENDER_EXTERNAL_URL
- [ ] Make main() async to properly await set_webhook
- [ ] Test webhook registration

## Command Debugging
- [ ] Check if commands are registered correctly
- [ ] Verify mention check in groups
- [ ] Add logging to command handlers

## Exception Handling
- [ ] Add try-except to all command handlers
- [ ] Add try-except to message handler
- [ ] Add try-except to wallet job

## Testing
- [ ] Test bot locally with polling mode
- [ ] Test webhook mode on Render
