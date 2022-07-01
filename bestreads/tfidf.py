"""
...
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from bestreads import text


def convert_df_dense_to_sparse(df_dense):
    """
    Utility function to convert dense DataFrames to sparse.

    Args:
        df_dense (pd.DataFrame): The dense DataFrame.

    Returns:
        df_sparse (pd.DataFrame): The sparse DataFrame.
    """
    return pd.DataFrame.sparse.from_spmatrix(
        sparse.csc_matrix(np.nan_to_num(df_dense.values)),
        columns=df_dense.columns, 
        index=df_dense.index
    )

class AbstractGenrePredictor(ABC):
    """The abstract base class for a book description genre predictor."""
    @abstractmethod
    def train(self):
        """Trains the predictor on the training data."""
    
    @abstractmethod
    def query(self, description):
        """Predicts the genre of a given description."""

class WeightedTFIDFGenrePredictor(AbstractGenrePredictor):
    '''
    '''
    def __init__(self, genre_and_votes: pd.Series, book_descriptions: pd.Series, n_genres: int = 10, verbose: bool = False):
        """
        Initializer.

        Args:
            genre_and_votes (pd.Series): The genre_and_votes column from the data.
            book_descriptions (pd.Series): The cleaned book descriptions.
            n_genres (int, optional): The number of genres to get from the data. 
                10 seems to get everything. Defaults to 10.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        self.genre_and_votes = genre_and_votes
        self.book_descriptions = book_descriptions
        self.verbose = verbose
    
        self.genre_and_votes_all = self._get_all_genres_and_votes(n=n_genres)
        self.genres_unique = self._compute_genres_unique()
        self.terms_unique = self._compute_terms_unique()
        self.term_to_index = {term: i for i,term in enumerate(self.terms_unique)}
        self.description_lengths = self._compute_book_description_lengths()
    
        # Computed by self.train():
        self.tf = None
        self.idf = None
        self.tfidf = None

    @property
    def ibooks(self):
        """indices of books"""
        return self.genre_and_votes.index
    
    @property
    def terms(self):
        """list of terms"""
        return self.terms_unique
    
    @property
    def genres(self):
        """list of genres"""
        return self.genres_unique
    
    @property
    def nbooks(self):
        """number of books"""
        return self.genre_and_votes.shape[0]
    
    @property
    def nterms(self):
        """number of terms"""
        return len(self.terms_unique)
    
    @property
    def ngenres(self):
        """number of genres"""
        return len(self.genres_unique)

    def _get_all_genres_and_votes(self, n=10):
        """
        Gets the votes for all genres for all books.

        Args:
            n (int, optional): The top number of genres to grab. Defaults to 10.

        Returns:
            genre_and_votes_all (pd.DataFrame): The DataFrame containing 
                'genres' and 'votes' columns, each containing a list of the 
                genres and votes, respectively.
        """
        genre_and_votes_all = self.genre_and_votes.apply(
            text._extract_genre_and_votes,
            n=n,
            reduce_subgenres=True
        )
        return pd.DataFrame(
            genre_and_votes_all.tolist(), 
            columns=['genres', 'votes'], 
            index=self.ibooks
        )
        
    def _compute_terms_unique(self):
        """
        Computes the sorted unique terms across all descriptions.

        Returns:
            terms_unique (list): The sorted list of unique terms.
        """
        terms_unique = set()
        for _,terms in self.book_descriptions.iteritems():
            terms_unique.update(terms)
        return sorted(terms_unique)
        
    def _compute_genres_unique(self):
        """
        Returns the set of genre names that were voted for (no duplicates).

        Args:
            genre_and_votes_all (pd.DataFrame): The DataFrame containing all of 
                the genres that were voted for.
            reduce_subgenres (bool): Will store a genre with a name like "Science
                Fiction - Aliens" as "Science Fiction" (Default: True)
                
        Returns:
            genres_unique (set): The set of genres that were voted for.
        """
        genre_matrix = np.vstack(self.genre_and_votes_all['genres'].apply(list))
        genres_df = pd.DataFrame(genre_matrix)
        genres_df = genres_df[genres_df.loc[:,0] != 'nan']

        genres_unique = set()
        for _,v in genres_df.iteritems():
            genres_unique.update(list(v.unique()))
        genres_unique.remove('nan')
        return sorted(genres_unique)
    
    def _compute_book_description_lengths(self):
        """
        llll

        Returns:
            _type_: _description_
        """
        df = self.book_descriptions.apply(len)
        df.name = 'description_lengths'
        return df

    def _get_genre_counts_all(self):
        """
        Counts the number of votes for each genre.

        Returns:
            genre_counts (pd.Series): The counts for each genre.
        """
        # Get counts for all votes from genres
        genre_counts = {genre: 0 for genre in self.genres_unique}
        for _,row in self.genre_and_votes_all.iterrows():
            for g,v in zip(*row):
                if isinstance(g, str):
                    genre_counts[g] += v
                    
        genre_counts = pd.Series(genre_counts)
        genre_counts = genre_counts.sort_values(ascending=False)
        genre_counts = genre_counts[genre_counts > 0]            
        return genre_counts
    
    def _get_raw_weights(self):
        '''
        '''
        df_raw_weights = pd.DataFrame(columns=self.genres)

        gviterator = self.genre_and_votes_all.iterrows()
        if self.verbose is True:
            print('Computing raw weights.')
            gviterator = tqdm(gviterator)

        for i,(genres,votes) in gviterator:
            total_votes = np.nansum(votes)
            votedict = {g: v/total_votes for g,v in zip(genres,votes)}
            df_raw_weights.loc[i] = votedict
        return convert_df_dense_to_sparse(df_raw_weights)
    
    def _get_weights(self):
        """
        Computes the "a" weight terms. These give the fractional contribution 
        to the weighted length of a genre by a book.

        Args:
            df_weights (_type_): _description_
            description_lengths (_type_): _description_

        Returns:
            _type_: _description_
        """
        df_raw_weights = self._get_raw_weights()
        if self.verbose is True:
            print('Computing weights.')
        Amat = (df_raw_weights.to_numpy().T * self.description_lengths.values).T
        Amat = Amat / np.nansum(Amat, axis=0)
        df_aweights = pd.DataFrame(Amat, columns=self.genres, index=self.ibooks)
        return convert_df_dense_to_sparse(df_aweights)
    
    def _get_tf_term_book(self):
        '''
        '''
        # Build the sparse COO matrix of tf(term,book)
        tf_tb_coo = []
        desc_iterator = self.book_descriptions.iteritems()
        if self.verbose is True:
            print('Computing tf(t,b).')
            desc_iterator = tqdm(desc_iterator)
        for ibook, desc in desc_iterator:
            length = len(desc)
            unique_terms = set(desc)
            # tf_dict = {term: desc.count(term)/length for term in unique_terms}
            for term in unique_terms:
                tf_tb_coo.append(
                    [ibook, self.term_to_index[term], desc.count(term)/length])
        ii, jj, values = zip(*tf_tb_coo)
        tf_tb_sparse = sparse.coo_matrix((values, (ii,jj)))
        
        # Build the sparse DataFrame
        df_tf_tb = pd.DataFrame.sparse.from_spmatrix(
            tf_tb_sparse, 
            columns=self.terms_unique
        )
        # Remove books with indices not included:
        df_tf_tb = df_tf_tb.iloc[self.ibooks]
        return df_tf_tb
    
    def compute_tf_terms_genres(self, df_weights, df_tf_tb):
        """
        _summary_

        Args:
            df_weights (_type_): _description_
            df_tf_tb (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.verbose is True:
            print('Computing tf(t,g).')
        mat = df_weights.sparse.to_coo().T @ df_tf_tb.sparse.to_coo()
        return pd.DataFrame.sparse.from_spmatrix(
            mat, 
            columns=self.terms, 
            index=self.genres
        )
        
    def _get_theta(self):
        # Build the sparse COO matrix of theta values
        theta_coo = []
        desc_iterator = self.book_descriptions.iteritems()
        if self.verbose is True:
            print('Computing theta.')
            desc_iterator = tqdm(desc_iterator)
        for ibook, desc in desc_iterator:
            unique_terms = set(desc)
            for term in unique_terms:
                if term in desc:
                    theta_coo.append([ibook, self.term_to_index[term], 1])
        ii, jj, values = zip(*theta_coo)
        theta_sparse = sparse.coo_matrix((values, (ii,jj)))

        # Build the sparse DataFrame
        df_theta = pd.DataFrame.sparse.from_spmatrix(
            theta_sparse, 
            columns=self.terms_unique
        )
        # Remove books with indices not included:
        df_theta = df_theta.iloc[self.ibooks]
        return df_theta
        
    def _get_n_t(self, df_weights, df_theta):
        """
        _summary_

        Args:
            df_weights (_type_): _description_
            df_theta (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.verbose is True:
            print('Computing n_t.')
        mat = df_weights.sparse.to_coo().T @ df_theta.sparse.to_coo()
        return pd.DataFrame.sparse.from_spmatrix(
            mat, 
            columns=self.terms, 
            index=self.genres
        )   
    
    def compute_idf(self, df_weights, df_theta):
        """
        _summary_

        Args:
            df_weights (_type_): _description_
            df_theta (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_t = self._get_n_t(df_weights, df_theta)
        # fill_value = np.log10(self.ngenres)
        if self.verbose is True:
            print('Computing idf.')
        mat = np.log10(self.ngenres / (1 + n_t.values))
        idf_dense = pd.DataFrame(mat, columns=self.terms, index=self.genres)
        return idf_dense
        
    def train(self):
        """
        _summary_
        """
        if self.verbose is True:
            print('Training weighted TF-IDF predictor.')
            print(f'  {self.nbooks} books')
            print(f'  {self.nterms} terms')
            print(f'  {self.ngenres} genres\n')
        # Compute the term frequency DataFrame: TF(term,genre)
        df_weights = self._get_weights()
        df_tf_tb = self._get_tf_term_book()
        self.tf = self.compute_tf_terms_genres(df_weights, df_tf_tb)
        
        # Compute the inverse document frequency DataFrame
        df_theta = self._get_theta()
        self.idf = self.compute_idf(df_weights, df_theta)
        
        tfidf_mat = self.tf.values * self.idf.values
        self.tfidf = pd.DataFrame.sparse.from_spmatrix(
            sparse.coo_matrix(tfidf_mat), 
            columns=self.terms, 
            index=self.genres
        )
        if self.verbose is True:
            print('Training done.')
    
    def query(self, description):
        if self.tfidf is None:
            raise ValueError('Predictor not trained yet. Call train() first.')
        
        tokenized_description = text.clean_text(description)
        description_terms = set(tokenized_description)
        # terms_present = [1 if term in description_terms else 0 
        #                  for term in self.terms_unique]
        # genre_scores = (self.tfidf * terms_present).sum(axis=1)
        genre_scores = self.tfidf[description_terms].sum(axis=1)
        genre_scores = genre_scores.sort_values(ascending=False)
        return genre_scores
