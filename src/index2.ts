import * as dotenv from "dotenv";
import wrap from "word-wrap";

import { OpenAI } from "langchain";
import { ConversationChain } from "langchain/chains";
import { BaseLLM } from "langchain/llms";

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

const num_turns = 5;

const solver = 'I will give you a problem to think about and you will try and solve it.' +
    ' Restate the problem, then think step by step and come up with the correct solution.' +
    ' I will respond to you and either point out issues or counterexamples, or tell you that you are correct.' +
    ' If you are correct, say "Problem solved."';
const critic = 'I will give you a problem and a potential solution.' +
    " You will find any issues or counterexamples that you can think of, or if I'm correct you will tell me so." +
    ' If I say "Problem solved." you will stop looking for issues and say "Problem solved." as well';

const c1 = await prime_chain(model, solver);
const c2 = await prime_chain(model, critic);

async function logres(person: string, res: string) {
  const wrappedRes = wrap(res, { indent: '  ', width: 100 });
  console.log(person);
  console.log(wrappedRes);
}

await logres('solver', c1.lastResponse);
await logres('critic', c2.lastResponse);

async function turn(chain1: Conv, chain2: Conv, conversation_starter?: string) {
  const res1 = await chain1.chain.call({ input: conversation_starter ?? chain2.lastResponse });
  const res2 = await chain2.chain.call({ input: res1.response });
  await logres('solver', res1.response);
  await logres('critic', res2.response);
  chain1.lastResponse = res1.response;
  chain2.lastResponse = res2.response;
}

let conversation_starter = 'People are arguing over whether large language models are just glorified autocomplete' +
    ' or whether they are displaying emergent properties of thought, world models, etc.' +
    ' What experiments would you run to find out which is true?';

for (let i = 0; i < num_turns; i += 1) {
  await turn(c1, c2, i === 0 ? conversation_starter : undefined);
}
console.log('=== Conversation complete ===');
