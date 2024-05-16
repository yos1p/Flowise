import { BaseMessage, ChainValues } from 'langchain/schema'
import { FlowiseMemory, ICommonObject, INode, INodeData, INodeParams, IUsedAgent } from '../../../src'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { GraphAgentExecutor } from '../../../src/graph'
import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
import { RunnableSequence } from '@langchain/core/runnables'
import { ToolsAgentStep } from 'langchain/dist/agents/openai_tools'
import { formatToOpenAIToolMessages } from 'langchain/agents/format_scratchpad/openai_tools'
import { AgentExecutor, ToolCallingAgentOutputParser } from '../../../src/agents'

class GraphAgent_Agents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    sessionId?: string
    badge?: string

    constructor(fields?: { sessionId?: string }) {
        this.label = 'Graph Agent'
        this.name = 'graphAgent'
        this.version = 1.0
        this.type = 'GraphAgentExecutor'
        this.category = 'Agents'
        this.icon = 'graphAgent.png'
        this.description = `Supervisor Agent that uses LangGraph to aggregate other Agents`
        this.baseClasses = [this.type]
        this.badge = 'NEW'
        this.inputs = [
            {
                label: 'Agents',
                name: 'agents',
                type: 'AgentExecutor',
                list: true
            },
            {
                label: 'Memory',
                name: 'memory',
                type: 'BaseChatMemory'
            },
            {
                label: 'Tool Calling Chat Model',
                name: 'model',
                type: 'BaseChatModel',
                description:
                    'Only compatible with models that are capable of function calling. ChatOpenAI, ChatMistral, ChatAnthropic, ChatVertexAI'
            },
            {
                label: 'Background & Persona',
                name: 'systemMessage',
                type: 'string',
                default: `You are a helpful AI Supervisor from ____ Company. You have several agents at your disposal.`,
                rows: 4,
                optional: true,
                additionalParams: true
            }
        ]
        this.sessionId = fields?.sessionId
    }

    async init(nodeData: INodeData, input: string, options: ICommonObject): Promise<any> {
        return prepareGraph(nodeData, options, { sessionId: this.sessionId, chatId: options.chatId, input })
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | ICommonObject> {
        const memory = nodeData.inputs?.memory as FlowiseMemory
        const isStreamable = options.socketIO && options.socketIOClientId
        const executor = await prepareGraph(nodeData, options, {
            sessionId: this.sessionId,
            chatId: options.chatId,
            input
        })

        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const callbacks = await additionalCallbacks(nodeData, options)

        let res: ChainValues = {}
        let sourceDocuments: ICommonObject[] = []
        let usedAgents: IUsedAgent[] = []

        if (isStreamable) {
            const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId)
            res = await executor.invoke({ input }, { callbacks: [loggerHandler, handler, ...callbacks] })
            if (res.usedAgents) {
                options.socketIO.to(options.socketIOClientId).emit('usedAgents', res.usedAgents)
                usedAgents = res.usedAgents
            }
        } else {
            res = await executor.invoke({ input }, { callbacks: [loggerHandler, ...callbacks] })
            if (res.usedAgents) {
                usedAgents = res.usedAgents
            }
        }

        let output = res?.output as string
        let finalRes = output

        await memory.addChatMessages(
            [
                {
                    text: input,
                    type: 'userMessage'
                },
                {
                    text: output,
                    type: 'apiMessage'
                }
            ],
            this.sessionId
        )

        if (sourceDocuments.length || usedAgents.length) {
            const finalRes: ICommonObject = { text: output }
            if (usedAgents.length) {
                finalRes.usedAgents = usedAgents
            }
            return finalRes
        }

        return finalRes
    }
}

const prepareSupervisor = async (
    nodeData: INodeData,
    _options: ICommonObject,
    flowObj: { sessionId?: string; chatId?: string; input?: string }
) => {
    const model = nodeData.inputs?.model as BaseChatModel
    const memory = nodeData.inputs?.memory as FlowiseMemory
    const maxIterations = nodeData.inputs?.maxIterations as string
    const systemMessage = nodeData.inputs?.systemMessage as string
    const nodeId = nodeData.inputs?.nodeId as string
    const memoryKey = memory.memoryKey ? memory.memoryKey : 'chat_history'
    const inputKey = memory.inputKey ? memory.inputKey : 'input'
    const agents = nodeData.inputs?.agents as AgentExecutor[]

    let supervisorPersona = `Background & Persona: ${systemMessage}`
    let supervisorMessage =
        supervisorPersona +
        `\n\nYour main task is to determine wich agent to use and pass them the message from User. Here are the available agents:`
    agents.forEach((agent) => {
        supervisorMessage += `\n- ${agent.nodeId}. ${agent.nodeFunction}`
    })
    supervisorMessage += `\n\nIf you find an agent that can help, start the message with: "Agent=AGENT_NAME;". 
                        Based on the chat history, you should try to get user's intention and find an agent that can help. 
                        Give simple, clear, and step-by-step instruction to the agent on what to do, based on the user's message and intention.
                        \n\nIf no agent is available, simply reply and ask for more information from user. 
                        \n\nREMEMBER: Never use AGENT_NAME that is not in the list.`

    const prompt = ChatPromptTemplate.fromMessages([
        ['system', supervisorMessage],
        new MessagesPlaceholder(memoryKey),
        ['human', `{${inputKey}}`],
        new MessagesPlaceholder('agent_scratchpad')
    ])

    const runnableAgent = RunnableSequence.from([
        {
            [inputKey]: (i: { input: string; steps: ToolsAgentStep[] }) => i.input,
            agent_scratchpad: (i: { input: string; steps: ToolsAgentStep[] }) => formatToOpenAIToolMessages(i.steps),
            [memoryKey]: async (_: { input: string; steps: ToolsAgentStep[] }) => {
                const messages = (await memory.getChatMessages(flowObj?.sessionId, true)) as BaseMessage[]
                return messages ?? []
            }
        },
        prompt,
        model,
        new ToolCallingAgentOutputParser()
    ])

    const executor = AgentExecutor.fromAgentAndTools({
        agent: runnableAgent,
        tools: [],
        sessionId: flowObj?.sessionId,
        chatId: flowObj?.chatId,
        input: flowObj?.input,
        verbose: process.env.DEBUG === 'true' ? true : false,
        maxIterations: maxIterations ? parseFloat(maxIterations) : undefined,
        nodeId
    })

    return executor
}

const prepareGraph = async (
    nodeData: INodeData,
    options: ICommonObject,
    flowObj: { sessionId?: string; chatId?: string; input?: string }
) => {
    const supervisor = await prepareSupervisor(nodeData, options, flowObj)
    const agents = nodeData.inputs?.agents as AgentExecutor[]
    return new GraphAgentExecutor({
        ...flowObj,
        supervisor,
        agents
    })
}

module.exports = { nodeClass: GraphAgent_Agents }
