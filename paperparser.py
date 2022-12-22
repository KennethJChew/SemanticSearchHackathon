class PaperParser():
    def __init__(self,corpus_text=None,model="small") -> None:
        """
        Model choices are:-
        1. large - length of embeddings per token is 4096
        2. small - length of embeddings per token is 1024
        3. multilingual-22-12
        """
        self.model = model
        self.models = ["small","large","multilingual-22-12"]
        self.client = cohere.Client("7lUDtMMSa1bVCEVIKEOms0jPImRnselfUQucOH5v")
        self.corpus = None
        # self.corpus_embeddings = co.embed(texts=corpus_text,model=model).embeddings
    

    def create_corpus(self,datafile,file_type="csv"):
        if type != "csv":
            data = pd.read_csv(datafile,sep="\t")
        else:
            data = pd.read_csv(datafile)
        corpus = {}
        corpus_records = data.to_dict(orient="records")

        for idx,record in enumerate(corpus_records):
            corpus[idx] = record
        self.corpus = corpus

    def get_corpus_embeddings(self):
        # indices are as follows:
        # 0 : dictionary index
        # 1 : Title
        # 2 : Language
        # 3 : Abstract
        # 4 : URL
        corpus_texts = []
        for idx in self.corpus:
            corpus_texts.append(self.corpus[idx]["ABSTRACT"])
        if len(corpus_texts) > 16:
            pass
        corpus_embeddings = self.client.embed(texts=corpus_texts,model=self.model).embeddings
        for idx,embedding in enumerate(corpus_embeddings):
            self.corpus[idx]["EMBEDDING"] = embedding
        

    def get_cos_sim(self,text: str) -> List[Tuple[str, float]]:
        """Return cosine similarity scores (sorted in descending order) of corpus documents with input text.

        Args:
            text (str): Input text to be compared against corpus.

        Returns:
            List[Tuple[str, float]]: Corpus documents and cosine similarity scores, sorted in descending order.
        """
        # Get embedding
        input_text = [text]
        text_embeddings = self.client.embed(texts=input_text,model=self.model).embeddings
        # Get cosine similarities
        res = []
        for record in self.corpus.items():
            cos_sim = dot(text_embeddings, record[1]["EMBEDDING"])/(norm(text_embeddings)*norm(record[1]["EMBEDDING"]))
            res.append((record[1]["TITLE"],float(cos_sim)))
        res.sort(key=lambda a:a[1],reverse=True)

        return res
    
    def calc_tf_idf(self,result="df",top_n=10):
        """
        Calculates the TF-IDF of the ABSTRACT field in the corpus and returns either
        the whole matrix or the top N results

        Args:
            result (str, optional): flag to toggle between returning the whole matrix or just the top n results.
                                    Set to 'df' to return whole matrix, set to 'top' to return the top n results. 
                                    Defaults to "df".
            top_n (int, optional):  number of results to return. Defaults to 5.

        Returns:
            _type_: _description_
        """
        tfidf_texts = []
        for idx in self.corpus:
            tfidf_texts.append(self.corpus[idx]["ABSTRACT"])
        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(tfidf_texts)
        dense = tf_idf.todense()
        dense_list = list(dense)
        output_features = vectorizer.get_feature_names_out()
        df = pd.DataFrame(dense, columns=output_features)

        if result == "top":
            top_df = []
            for row,index in df.iterrows():
                # print(index.sort_values(ascending=False)[:top_n])
                top_df.append(index.sort_values(ascending=False)[:top_n])
            
            return pd.DataFrame(top_df)
        else:
            return df