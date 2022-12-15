import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ONES;

public class Iterator {

    public static void main(String[] args) {
        DataSetIterator baseIterator = new RandomDataSetIterator(4, new long[]{1, 4}, new long[]{1, 4}, ONES, ONES);
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(baseIterator, 4, 0.5);
        List<DataSetIterator> iterators = splitter.getIterators(); // throws exception
        DataSetIterator iterator0 = iterators.get(0);
        DataSetIterator iterator1 = iterators.get(1);
    }
}
