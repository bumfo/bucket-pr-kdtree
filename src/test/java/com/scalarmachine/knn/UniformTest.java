package com.scalarmachine.knn;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;
import jk.tree.FloatKDTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

public final class UniformTest {
    public static final int DIM = 10;
    public static final int N = 100000;
    public static final int K = 200;
    public static final int RECALL_K = K;
    public static final int TRIES = 3;

    public static void main(String[] args) {
        ArrayList<Word> words = getWords();

        FloatKDTree<Word> tree = getTree(words);
        HnswIndex<Integer, float[], Word, Float> hnswIndex = getHnsw(words);

        Collections.shuffle(words, new Random(1234));

        int good = 0;
        int bad = 0;
        for (int i = 0; i < 1000; i++) {
            Word word = words.get(i);
            Set<Integer> set = new HashSet<>();
            // for (FloatKDTree.SearchResult<Word> res : tree.nearestNeighbours(word.vector(), K)) {
            //     set.add(res.payload.id());
            // }
            for (SearchResult<Word, Float> res : hnswIndex.findNearest(word.vector(), K)) {
                set.add(res.item().id());
            }
            for (FloatKDTree.SearchResult<Word> res : tree.nearestNeighbours(word.vector(), RECALL_K)) {
                if (set.contains(res.payload.id())) {
                    good += 1;
                } else {
                    bad += 1;
                }
            }
        }
        System.out.println("good bad = " + good + ", " + bad);
        System.out.println("recall@" + RECALL_K + " = " + (good * 1.0 / (good + bad)));

        for (int i = 0; i < TRIES; i++) {
            queryKDTree(words, tree);
        }

        for (int i = 0; i < TRIES; i++) {
            queryHnsw(words, hnswIndex);
        }
    }

    private static void queryHnsw(ArrayList<Word> words, HnswIndex<Integer, float[], Word, Float> hnswIndex) {
        long start;
        long duration;
        long end;
        start = System.currentTimeMillis();

        for (int i = 0; i < 1000; i++) {
            Word word = words.get(i);
            hnswIndex.findNearest(word.vector(), K);
        }

        end = System.currentTimeMillis();
        duration = end - start;

        System.out.printf("HNSW lookup %d\n", duration);
    }

    private static void queryKDTree(ArrayList<Word> words, FloatKDTree<Word> tree) {
        long start;
        long end;
        long duration;
        start = System.currentTimeMillis();

        for (int i = 0; i < 1000; i++) {
            Word word = words.get(i);
            tree.nearestNeighbours(word.vector(), K);
        }

        end = System.currentTimeMillis();
        duration = end - start;

        System.out.printf("KDTree lookup %d\n", duration);
    }

    private static HnswIndex<Integer, float[], Word, Float> getHnsw(ArrayList<Word> words) {
        long start;
        long end;
        long duration;
        HnswIndex<Integer, float[], Word, Float> hnswIndex = HnswIndex
            .newBuilder(DIM, DistanceFunctions.FLOAT_MANHATTAN_DISTANCE, words.size())
            .withM(8)
            .withEf(200)
            .withEfConstruction(32)
            .build();

        start = System.currentTimeMillis();

        // hnswIndex.addAll(words, (workDone, max) -> System.out.printf("Added %d out of %d words to the index.%n", workDone, max));
        for (Word word : words) {
            hnswIndex.add(word);
        }

        end = System.currentTimeMillis();
        duration = end - start;

        System.out.printf("Creating index with %d words took %d millis which is %d seconds.%n", hnswIndex.size(), duration, MILLISECONDS.toSeconds(duration));
        return hnswIndex;
    }

    private static FloatKDTree<Word> getTree(ArrayList<Word> words) {
        long duration;
        long end;
        long start;
        FloatKDTree<Word> tree = new FloatKDTree.Manhattan<>(DIM);

        start = System.currentTimeMillis();

        for (Word word : words) {
            tree.addPoint(word.vector(), word);
        }

        end = System.currentTimeMillis();
        duration = end - start;

        System.out.printf("Creating index with %d words took %d millis which is %d seconds.%n", tree.size(), duration, MILLISECONDS.toSeconds(duration));
        return tree;
    }

    private static ArrayList<Word> getWords() {
        ArrayList<Word> words = new ArrayList<>(N);
        for (int i = 0; i < N; i++) {
            float[] vector = new float[DIM];
            for (int j = 0; j < DIM; j++) {
                vector[j] = (float) Math.random();
            }
            words.add(new Word(i, vector));
        }
        return words;
    }

    public static final class Word implements Item<Integer, float[]> {
        private static final long serialVersionUID = 1L;

        private final Integer id;
        private final float[] vector;

        public Word(Integer id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }

        @Override
        public Integer id() {
            return id;
        }

        @Override
        public float[] vector() {
            return vector;
        }

        @Override
        public int dimensions() {
            return vector.length;
        }

        @Override
        public String toString() {
            return "Word{" +
                "id='" + id + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
        }
    }
}
