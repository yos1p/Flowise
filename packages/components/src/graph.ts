import { END, MessageGraph } from '@langchain/langgraph'
import { AgentExecutor } from './agents'
import { BaseChain, ChainInputs } from 'langchain/chains'
import { CallbackManagerForChainRun } from '@langchain/core/callbacks/manager'
import { RunnableConfig } from '@langchain/core/runnables'
import { StringMomentoTokenProvider } from '@gomomento/sdk'
import { ChainValues } from 'langchain/schema'

interface GraphAgentInput extends ChainInputs {
    supervisor: AgentExecutor
    agents?: Array<AgentExecutor>
}

type GraphAgentOutput = ChainValues

export class GraphAgentExecutor extends BaseChain<ChainValues, GraphAgentOutput> {
    agents: Array<AgentExecutor> = []

    graph: any

    sessionId?: string

    chatId?: string

    input?: StringMomentoTokenProvider

    constructor(input: GraphAgentInput & { sessionId?: string; chatId?: string; input?: string; isXML?: boolean }) {
        super(input)
        if (input.agents) this.agents = input.agents
        this.sessionId = input.sessionId
        this.chatId = input.chatId

        const graph = new MessageGraph()
        const supervisor = input.supervisor
        graph.addNode('supervisor', async (state: ChainValues[]) => {
            return supervisor.invoke(state[0])
        })
        this.agents.forEach((agent) => {
            this.addNextAgent(graph, agent, agent.nextAgent)
        })
        // When the supervisor returns, route to the agent identified in the supervisor's output
        graph.addConditionalEdges('supervisor', (state: ChainValues[]) => {
            const output = state[state.length - 1].output

            // if output contains: Agent=AGENT_NAME;, then route to AGENT_NAME
            if (output.includes('Agent=')) {
                // Extract the agent name from the output
                const agentName = output.split('Agent=')[1].split(';')[0]
                return agentName
            }
            return END
        })

        graph.setEntryPoint('supervisor')
        this.graph = graph.compile()
    }

    addNextAgent(graph: MessageGraph<any>, agent: AgentExecutor, nextAgent?: AgentExecutor) {
        if (!agent.nodeId) throw new Error('Agent must have a nodeId')

        try {
            graph.addNode(agent.nodeId, async (state: ChainValues[]) => {
                // let supervisorMessage = state[1].output
                // if (supervisorMessage.includes('Agent=' + agent.nodeId))
                //     supervisorMessage = supervisorMessage.replace('Agent=' + agent.nodeId + ';', '')

                // const inputMsg = `User's message: ${state[0].input}
                // \n\nAdditional instruction from previous AI Assistant:
                // \n${supervisorMessage}`
                const inputMsg = state[0].input
                return agent?.invoke({
                    input: inputMsg,
                    sessionId: this.sessionId
                })
            })
        } catch (e) {
            // If the node already added previously, we ignore
        }

        if (nextAgent) {
            if (!nextAgent.nodeId) throw new Error('Agent must have a nodeId')

            // It is possible the node already added previously, so if error, we ignore
            try {
                graph.addNode(nextAgent.nodeId, async (state: ChainValues[]) => {
                    let inputMsg
                    if (state.length > 1) {
                        inputMsg = state[state.length - 1].output
                    } else {
                        inputMsg = state[0].input
                    }
                    return nextAgent?.invoke({
                        input: inputMsg,
                        sessionId: this.sessionId
                    })
                })
            } catch (e) {
                // If the node already added previously, we ignore
            }

            try {
                const agentId = agent.nodeId
                const nextAgentId = nextAgent.nodeId
                graph.addEdge(agentId, nextAgentId)
                this.addNextAgent(graph, nextAgent, nextAgent.nextAgent)
            } catch (e) {
                // It is possible the nextAgent already added previously, so if error, we ignore
            }
        } else {
            // Last node always report to the supervisor
            graph.addEdge(agent.nodeId, END)
        }
    }

    _call(
        values: ChainValues,
        _runManager?: CallbackManagerForChainRun | undefined,
        _config?: RunnableConfig | undefined
    ): Promise<GraphAgentOutput> {
        return new Promise((resolve, reject) => {
            this.graph
                .invoke(values)
                .then((res: any) => {
                    // Check if res is an array, if so, get the latest value
                    // If so, save it to finalOutput variable
                    if (Array.isArray(res)) {
                        res = res[res.length - 1]
                    }
                    resolve(res)
                })
                .catch((error: any) => {
                    reject(error)
                })
        })
    }

    _chainType(): string {
        return 'graph_executor' as const
    }

    get inputKeys() {
        return []
    }

    get outputKeys() {
        return []
    }
}
