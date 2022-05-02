import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class Loss {

    public static void main(String[] args) throws IOException, InterruptedException {

        SameDiff sd = SameDiff.create();

        int batchSize = 4;
        int seqLength = 8;

        SDVariable features = sd.squeeze(sd.placeHolder("features", FLOAT, batchSize, 1, seqLength), 1);
        SDVariable labels = sd.squeeze(sd.placeHolder("labels", FLOAT, batchSize, 1, seqLength), 1);
        SDVariable bias = sd.var("bias", new ZeroInitScheme('c'), FLOAT, seqLength);
        SDVariable predictions = features.add("predictions", bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.01))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        // task: add 1 to the inputs
        SequenceRecordReader featureReader = new CSVLineSequenceRecordReader();
        featureReader.initialize(new FileSplit(new File("src/main/resources/features.csv")));
        SequenceRecordReader labelReader = new CSVLineSequenceRecordReader();
        labelReader.initialize(new FileSplit(new File("src/main/resources/labels.csv")));
        DataSetIterator iterator = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, batchSize, 0, true);

        // ScoreListener will consistently report a loss of 0
        History hist = sd.fit(iterator, 100, new ScoreListener(1));

        // The recorded loss values are also constantly 0
        LossCurve curve = hist.lossCurve();
        System.out.println("Loss curve:");
        System.out.println(curve.getLossNames());
        System.out.println(curve.getLossValues());

        // However, the loss calculated here is > 0 and the values of bias are far from the optimal value of 1
        Map<String, INDArray> map = sd.output(iterator, "loss", "bias");
        System.out.println("Final loss: " + map.get("loss"));
        System.out.println("Final bias: " + map.get("bias"));
    }
}
