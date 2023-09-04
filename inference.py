import requests
import os

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

prompt = """
<Human>You are a legal professional tasked with drafting a Non-Disclosure Agreement (NDA).
Below is an instruction that describes a specific scenario. Your objective is to create
a comprehensive NDA based on the given context. Your NDA should address the key points provided.
After drafting the NDA, pair it with a response that properly completes the request.

### Instruction:
You are representing the company "Tech Innovators Inc." which is disclosing proprietary information to an individual consultant, "Jane Smith," for a top-secret project related to cutting-edge technology. The project's code name is "Project Quantum." The NDA should cover the following points:

1. Definition of Confidential Information: Clearly define what constitutes confidential information in the context of "Project Quantum."

2. Confidentiality Obligations: Outline the recipient's responsibilities in protecting the confidential information and preventing its disclosure.

3. Permitted Use: Specify the authorized purposes for which the recipient can use the confidential information.

4. Non-Disclosure: Clearly state that the recipient must not disclose or share the confidential information with any third party without explicit written consent.

5. Duration of Agreement: Specify the duration for which the NDA will be in effect.

6. Remedies for Breach: Describe the actions that Tech Innovators Inc. can take in case of a breach of the NDA.

Draft a complete Non-Disclosure Agreement based on the above scenario and instructions. ALSO JUST WRITE THE NDA DRAFT IN A WAY THAT YOU DO NOT GREET IN THE START OR THE END

<Assistant>

"""
endpoint = 'https://api.together.xyz/inference'

res = requests.post(endpoint, json={
    "model": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    "prompt": prompt,
    "top_p": 1,
    "top_k": 40,
    "temperature": 0.8,
    "max_tokens": 4000,
    "repetition_penalty": 1,
    "stop": "\n\n"
}, headers={
    "Authorization": "Bearer 1b1af0edea0ead6ed330057bd1f9cc3e0bae43a0ead2031b19eb45e8f95fa9b9",
    "User-Agent": "1b1af0edea0ead6ed330057bd1f9cc3e0bae43a0ead2031b19eb45e8f95fa9b9"
})
print(res.json())
# print(res.json()['output']['choices'][0]['text'])