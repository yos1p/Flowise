import express from 'express'
import voiceapiController from '../../controllers/voiceapi'
const router = express.Router()

router.post('/:chatflowId/chat/completions', voiceapiController.chatCompletions)

export default router
