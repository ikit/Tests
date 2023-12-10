
// Etape 1 Récupération donnée et formatage csv
import axios from "axios";
import { select } from "xpath"
import { DOMParser } from "@xmldom/xmldom";
import * as csv2json from "json-2-csv";
import * as fs from "fs";
// Etape 2 - Embedding
import { ChromaClient, Collection, IEmbeddingFunction, OpenAIEmbeddingFunction } from 'chromadb' // DB pour s'interfacer avec un LLM
// import { HuggingFaceTransformersEmbeddings } from "langchain/embeddings/hf_transformers"; // API pour utiliser des models pré-entrainé d'IA comme HuggingFace
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { Chroma } from "langchain/vectorstores/chroma";
// Etape 3 - LLM
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";

const OAIK = "clés d'api OpenAI";

export interface EncycloData {
    name: string;
    url: string;
    text: string;
}



// Récupération d'une base de donnée
// wiki Zelda: https://www.puissance-zelda.com/encyclopedie
async function getZeldaEncyclopedia(): Promise<EncycloData[]> {
    const htmlPage = await axios.get("https://www.puissance-zelda.com/encyclopedie");
    var xml = htmlPage.data;
    var doc = new DOMParser({errorHandler: function(){}}).parseFromString(xml, 'text/xml');
    var nodes = select("//ul[@class=\"lettreEncyclo\"]/li/a/@href", doc) as any[];
    
    const db:string[] = nodes.map(n => n.nodeValue);

    // A partir des liens du sommaire, on récupère les artcicles
    async function getText(url: string) {
        const htmlPage = await axios.get(url);
        var xml = htmlPage.data;
        var doc = new DOMParser({errorHandler: function(){}}).parseFromString(xml, 'text/xml');
        const res = select("//div[@class=\"col-12 mb-lg-3\"]/text()", doc);
        return res.toString().replace( /[\r\n]+/gm, "" ).replace( /[\t]+/gm, "" );
    }

    const res = [];
    let count = 0;
    for (const url of db) {
        count += 1;
        process.stdout.write(`\rRécupération du contenu: ${count}/${db.length}`);
        res.push({ 
            name: url.substring(49),
            url: url as string,
            text: await getText(url)
        });
    }
    console.log("")
    return res;
}

// On sauvegarde en local dans un fichier csv
async function encycloToCsv(data: EncycloData[], filename: string) {
    fs.writeFileSync(filename, csv2json.json2csv(data));
}
function getEncycloFromCsv(filename: string): EncycloData[]  {
    const csv = csv2json.csv2json(fs.readFileSync(filename, { encoding:"utf8"})) as EncycloData[];
    return csv;
}

async function getOrCreateChromaDb(csvFile: string): Promise<Collection> {
    console.log("getOrCreateChromaDb: " + csvFile);
    // TODO: réussir à utiliser ce putain module js de merde : https://www.npmjs.com/package/@xenova/transformers
    // const pipeline = await dynamicImport("@xenova/transformers");
    // const pipeline = await import("@xenova/transformers"); // API pour utiliser des models pré-entrainé d'IA comme HuggingFace
    // Comme on ne veut pas utiliser OpenAI, on va se créer notre propre méthode d'embedding basé sur les modèles pré-entrainé libre de HF
    class MyEmbeddingFunction implements IEmbeddingFunction {
        private pipe: any = null;
      
        public async generate(texts: string[]): Promise<number[][]> {
            // On choisi un model de vectorisation pour l'embedding: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            // On choisi un modèle qui donne la similitude entre différents textes
            // (pour pouvoir dire quel article correspond le plus à ce qu'on cherche)

            // dynamic import de la lib transformers car c'est de la merde js
            console.log("== > start import")
           await import("@xenova/transformers")
                .then(async module => {
                    this.pipe = await module.pipeline.call("sentence-similarity", "sentence-transformers/all-MiniLM-L6-v2");
                    console.log("== > end import")
                })
                .catch(error => console.log('Import dynamique de la lib transformers a échoué:' + error.message));

            // do things to turn texts into embeddings with an api_key perhaps
            console.log("== > return import.pipe")
            return this.pipe(texts);
        }
    }
    
    const data = getEncycloFromCsv(csvFile);

    // On crée la collection ChromaDB
    const client = new ChromaClient();

    // On supprime la base si elle existe déjà pour assurer que l'étape de création fonctionne
    const collections = await client.listCollections()
    const col = collections.find(c => c.name === csvFile);
    if (col) {
        console.log(" > Suppression de la base " + col.name);
        await client.deleteCollection({ name: col.name });
    }
    

    // TODO: using custom Embedding
    console.log(" > Création de la base " + col.name);
    const collection = await client.createCollection({ 
        name: csvFile,
        embeddingFunction: new OpenAIEmbeddingFunction({ 
            openai_api_key: OAIK,
        })
    });
    // On y ajoute nos données persos
    console.log(" > Ajout des donnée à la base Chroma");
    await collection.add({
        ids: data.map(e => e.name),
        metadatas: data.map(e => ({ source: `${e.name}: ${e.url}`})),
        documents: data.map(e => e.text),
    });

    return collection;
}


async function getVectoreStore(chromaDb: Collection) {
    console.log("chromaDb vectorisation for OpenAI")
    // On 
    // const model = new HuggingFaceTransformersEmbeddings({
    //     modelName: "Xenova/all-MiniLM-L6-v2",
    // });

    console.log(" > Vectorisation de la collection selon le transformer d'OpenIA");
    const vectorStore = await Chroma.fromExistingCollection(
        new OpenAIEmbeddings({ openAIApiKey: OAIK }),
        { collectionName: chromaDb.name },
    );

    return vectorStore;
}


async function chatBot(vectorStore: Chroma) {
    console.log("Création du LLM")
    // On défini le LLM à utiliser
    // TODO: utiliser un modèle libre récupéré via HF
    console.log(" > Récupération du model OpenAI GPT3.5-turbo");
    const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", openAIApiKey: OAIK });

    // On crée le template du prompt chatBot pour conditionner l'IA et lui donner le context "data" pour répondre les questions de l'utilisateur
    console.log(" > Création du template");
    const template = `Tu es un assitant qui parle français et qui peut peut répondre aux questionssur l'encyclopédie de Zelda. 
    Utilise seulement le contexte et répond aux questions en Français. 
    Si tu ne trouve pas la réponse dans le contexte, répond que l'information ne se trouve pas dans l'Encyclopédie Zelda.
    Contexte : {context}
    Question : {question}`;

    // On utlise langchain pour "brancher le LLM (chatBot) avec notre base de donnée vectorisée"
    console.log(" > Chainage du vecteur et LLM avec langchain");
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
        prompt: PromptTemplate.fromTemplate(template),
        returnSourceDocuments: true
    });

    return chain;
}


async function main(query: string) {
    console.log("Ma question: " + query,"\n---");

    //
    // Etape 1 - Récupération/Nettoyage d'un jeu de donnée perso
    //
    // // Création d'un jeu de donnée (wiki zelda)
    // const csv = await getZeldaEncyclopedia()
    // encycloToCsv(csv, "zelda.csv")
    // // Check 
    // console.log(getEncycloFromCsv("zelda.csv"));

    // 
    // Etape 2 - "Embedding": conversion du jeu de donnée perso dans en un modèle compréhensible pour les LLM avec Chroma
    //
    // https://js.langchain.com/docs/integrations/vectorstores/chroma

    const collection = await getOrCreateChromaDb("zelda.csv");
    // Requêtage simple
    // const res1 = await collection.query({
    //     nResults: 2,
    //     queryTexts: ["Abeille"],
    // });
    // console.log(res1)
    // requêtage via langchain retrievers
    const vectorStore = await getVectoreStore(collection);
    //console.log("\n\n====\n\n", vectorStore.similaritySearch("Abeille", 2), "\n\n")
    

    //
    // Etape 3 - "ChatBot Q&A": intégration avec un chatbot pour l'interprétation des questions en langage naturel
    //
    // https://js.langchain.com/docs/use_cases/question_answering/
    const chat = await chatBot(vectorStore);
    const res = await chat.call({ query })
    console.log("---\nRéponse de l'IA:\n", res.text, "\nSources:");
    for (const r of res.sourceDocuments) {
        console.log(" - " + r.metadata.source);
    }
}

// main("A quoi servent les abeilles ?");
main("Comment s'appelle la princesse ?");
