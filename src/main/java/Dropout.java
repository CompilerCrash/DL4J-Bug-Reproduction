import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.Collections;

public class Dropout {

    private static final INDArray in = Nd4j.ones(4, 8);

    public static void main(String[] args) {
        String separator = "=".repeat(100);

        dropoutZero();
        System.out.println(separator);
        dropoutInverted();
        System.out.println(separator);
        dropoutValues();
        System.out.println(separator);
        dropoutBackprop();
    }

    public static void dropoutZero() {
        INDArray res = Nd4j.nn.dropout(in, false, 0.0); // throws exception
        System.out.println(res);
    }

    public static void dropoutInverted() {
        INDArray res1 = Nd4j.nn.dropout(in, false, 1.0);
        System.out.println(res1); // same as res0 but should be different
        INDArray res0 = Nd4j.nn.dropout(in, true, 1.0);
        System.out.println(res0); // same as res1 but should be different
    }

    public static void dropoutValues() {
        INDArray res02 = Nd4j.nn.dropout(in, true, 0.2);
        System.out.println(res02); // should only contain 1s and 0s but contains 5s instead of 1s
        INDArray res08 = Nd4j.nn.dropout(in, true, 0.8);
        System.out.println(res08); // should only contain 1s and 0s but contains 1.25s instead of 1s
    }

    public static void dropoutBackprop() {
        int batchSize = 4;
        int seqLength = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, seqLength);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, seqLength);
        SDVariable predictions = sd.nn.dropout("predictions", features, false, 0.5);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        RecordReader reader = new CollectionRecordReader(
                Collections.nCopies(batchSize, Collections.nCopies(2 * seqLength, new IntWritable(1))));
        DataSetIterator iterator = new RecordReaderDataSetIterator(
                reader, batchSize, seqLength, 2 * seqLength - 1, true);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }
}
