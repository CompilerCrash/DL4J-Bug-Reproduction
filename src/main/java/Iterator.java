import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Collections;
import java.util.List;

public class Iterator {

    public static void main(String[] args) {
        RecordReader reader = new CollectionRecordReader(Collections.nCopies(4, Collections.nCopies(4, new IntWritable(1))));
        DataSetIterator baseIterator = new RecordReaderDataSetIterator(reader, 1, 3, 3, true);
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(baseIterator, 4, 0.5);
        List<DataSetIterator> iterators = splitter.getIterators(); // throws exception
        DataSetIterator iterator0 = iterators.get(0);
        DataSetIterator iterator1 = iterators.get(1);
    }
}
