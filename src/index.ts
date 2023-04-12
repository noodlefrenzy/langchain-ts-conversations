import * as dotenv from "dotenv";
import wrap from "word-wrap";

import { OpenAI } from "langchain";
import { ConversationChain } from "langchain/chains";
import { BaseLLM } from "langchain/llms";
import { PromptTemplate } from "langchain/prompts";

dotenv.config();

class Conv {
  chain: ConversationChain;
  
  lastResponse: string;

  constructor(chain: ConversationChain, lastResponse: string) {
    this.chain = chain;
    this.lastResponse = lastResponse;
  }
}

const model = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0.8
});

async function prime_chain(model: BaseLLM, initial_prompt: string) {
  const chain = new ConversationChain({ llm: model });
  return new Conv(chain, (await chain.call({ input: initial_prompt })).response);
}

const person1 = { person: 'Noam Chomsky', details: 'renowned linguist' };
const person2 = { person: 'Yann LeCun', details: 'one of the creators of Deep Learning' };
const num_turns = 5;

const initial_prompt = new PromptTemplate({ template: "Please converse with me as if you are {person}, {details}. Say 'yes' if you agree.", inputVariables: ['person', 'details'] });

const c1 = await prime_chain(model, await initial_prompt.format(person1));
const c2 = await prime_chain(model, await initial_prompt.format(person2));

async function logres(person: string, res: string) {
  const wrappedRes = wrap(res, { indent: '  ', width: 100 });
  console.log(person);
  console.log(wrappedRes);
}

await logres(person1.person, c1.lastResponse);
await logres(person2.person, c2.lastResponse);

async function turn(chain1: Conv, chain2: Conv, conversation_starter?: string) {
  const res1 = await chain1.chain.call({ input: conversation_starter ?? chain2.lastResponse });
  const res2 = await chain2.chain.call({ input: res1.response });
  await logres(person1.person, res1.response);
  await logres(person2.person, res2.response);
  // eslint-disable-next-line no-param-reassign
  chain1.lastResponse = res1.response;
  // eslint-disable-next-line no-param-reassign
  chain2.lastResponse = res2.response;
}

const conversation_starter = 'People are arguing over whether large language models are just glorified autocomplete or whether they are displaying emergent properties of thought, world models, etc. What do you think about it?';
for (let i = 0; i < num_turns; i += 1) {
  await turn(c1, c2, i === 0 ? conversation_starter : undefined);
}
console.log('=== Conversation complete ===');
