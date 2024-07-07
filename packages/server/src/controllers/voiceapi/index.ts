import { Request, Response, NextFunction } from 'express'

type VAPI_Message = {
    messages: Message[]
}
type Message = {
    role: string
    content: string
}

const chatCompletions = async (req: Request, res: Response, next: NextFunction) => {
    try {
        // Access the JSON body sent by the client
        const data = req.body
        // Function to get the latest message with the role as 'user'
        const getLatestUserMessage = (data: VAPI_Message) => {
            // Filter messages with role 'user'
            const userMessages = data.messages.filter((msg) => msg.role === 'user')
            // Get the last message from the filtered list
            const latestUserMessage = userMessages[userMessages.length - 1]
            return latestUserMessage.content
        }
        const userMessage = getLatestUserMessage(data)

        // Flowise SessionID
        const storedSessionId = data.metadata.sessionId
        const phoneNumber =
            data.customer == undefined || data.customer == null || data.customer.number == undefined || data.customer.number == null
                ? 'NOT AVAILABLE'
                : data.customer.number
        // Get current date in dd-MM-yyyy
        const currentDate = new Date().toLocaleDateString('en-GB')

        const BASE_URL = `${req.protocol}://${req.headers.host}` // Constructs the full URL
        const chatflowId = req.params.chatflowId
        const API_URL = BASE_URL + '/api/v1/prediction/' + chatflowId
        const headers = { 'Content-Type': 'application/json', Authorization: '' }
        if (req.headers.authorization != undefined && req.headers.authorization != null) {
            headers['Authorization'] = req.headers.authorization
        }
        const response = await fetch(API_URL, {
            method: 'POST',
            body: JSON.stringify({
                // The format should always include metadata such as TODAY, current phone number, etc.
                question:
                    "{metadata: {today: '" +
                    currentDate +
                    "', currentPhoneNumber: '" +
                    phoneNumber +
                    "'}, userMessage: '" +
                    userMessage +
                    "'}",
                overrideConfig: {
                    sessionId: storedSessionId
                }
            }),
            headers: headers
        })
        const result = await response.json()
        const aiResponse = result.text

        res.setHeader('Content-Type', 'text/event-stream')
        res.setHeader('Cache-Control', 'no-cache')
        res.setHeader('Connection', 'keep-alive')

        const openAICompletions = {
            id: 'chatcmpl-123',
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: 'gpt-4o',
            system_fingerprint: null,
            choices: [
                {
                    index: 0,
                    delta: { role: 'assistant', content: aiResponse },
                    logprobs: null,
                    finish_reason: 'stop'
                }
            ]
        }
        const openAIStopCompletions = {
            id: 'chatcmpl-123',
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: 'gpt-4o',
            system_fingerprint: null,
            choices: [
                {
                    index: 0,
                    delta: {},
                    logprobs: null,
                    finish_reason: 'stop'
                }
            ]
        }

        res.write('data: ' + JSON.stringify(openAICompletions) + '\n\n')
        res.write('data: ' + JSON.stringify(openAIStopCompletions) + '\n\n')
        res.write('data: [DONE]\n\n')
        res.end()
    } catch (error) {
        next(error)
    }
}

export default {
    chatCompletions
}
