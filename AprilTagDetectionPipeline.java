package org.firstinspires.ftc.teamcode.TeleOp;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.openftc.apriltag.AprilTagDetection;
import org.openftc.apriltag.AprilTagDetectorJNI;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

class AprilTagDetectionPipeline extends OpenCvPipeline
{
    private Map<Integer, Double>tagDistances = new HashMap<>();
    private Set<Integer> currentTagIDs = new HashSet<>();

    public boolean tagfaund = false;
    private long nativeApriltagPtr;
    private Mat grey = new Mat();
    private ArrayList<AprilTagDetection> detections = new ArrayList<>();

    private ArrayList<AprilTagDetection> detectionsUpdate = new ArrayList<>();
    private final Object detectionsUpdateSync = new Object();

    public ArrayList<AprilTagDetection> getDetections() {
        return detections;
    }

    Mat cameraMatrix;

    Scalar blue = new Scalar(7,197,235,255);
    Scalar red = new Scalar(255,0,0,255);
    Scalar green = new Scalar(0,255,0,255);
    Scalar white = new Scalar(255,255,255,255);

    double fx;
    double fy;
    double cx;
    double cy;

    // UNITS ARE METERS
    double tagsize;
    double tagsizeX;
    double tagsizeY;

    private float decimation;
    private boolean needToSetDecimation;
    private final Object decimationSync = new Object();

    private Telemetry telemetry;  // Telemetry nesnesini tanımlayın.

    public AprilTagDetectionPipeline(double tagsize, double fx, double fy, double cx, double cy, Telemetry telemetry)
    {
        this.tagsize = tagsize;
        this.tagsizeX = tagsize;
        this.tagsizeY = tagsize;
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;

        this.telemetry = telemetry;  // Telemetry nesnesini kurucuya ekleyin.

        constructMatrix();

        // Allocate a native context object. See the corresponding deletion in the finalizer
        nativeApriltagPtr = AprilTagDetectorJNI.createApriltagDetector(AprilTagDetectorJNI.TagFamily.TAG_36h11.string, 3, 3);
    }

    @Override
    public void finalize()
    {
        // Might be null if createApriltagDetector() threw an exception
        if(nativeApriltagPtr != 0)
        {
            // Delete the native context we created in the constructor
            AprilTagDetectorJNI.releaseApriltagDetector(nativeApriltagPtr);
            nativeApriltagPtr = 0;
        }
        else
        {
            System.out.println("AprilTagDetectionPipeline.finalize(): nativeApriltagPtr was NULL");
        }
    }

    @Override
    public Mat processFrame(Mat input)
    {
        // Convert to greyscale
        Imgproc.cvtColor(input, grey, Imgproc.COLOR_RGBA2GRAY);

        synchronized (decimationSync)
        {
            if (needToSetDecimation)
            {
                AprilTagDetectorJNI.setApriltagDetectorDecimation(nativeApriltagPtr, decimation);
                needToSetDecimation = false;
            }
        }

        // Run AprilTag
        detections = AprilTagDetectorJNI.runAprilTagDetectorSimple(nativeApriltagPtr, grey, tagsize, fx, fy, cx, cy);

        synchronized (detectionsUpdateSync)
        {
            detectionsUpdate = detections;
        }

        // Clear current tag IDs
        currentTagIDs.clear();

        // Track detected tags and their distances
        for (AprilTagDetection detection : detections)
        {
            // Draw the ID of the detection in red
            Imgproc.putText(input, "ID: " + detection.id, detection.corners[0], Imgproc.FONT_HERSHEY_SIMPLEX, 1, red, 3);

            if (detection.id >= 0) {
                tagfaund = true;
            }

            // Compute the pose
            Pose pose = poseFromTrapezoid(detection.corners, cameraMatrix, tagsizeX, tagsizeY);

            // Calculate the distance to the tag
            double distance = Math.sqrt(
                    Math.pow(pose.tvec.get(0, 0)[0], 2) +
                            Math.pow(pose.tvec.get(1, 0)[0], 2) +
                            Math.pow(pose.tvec.get(2, 0)[0], 2)
            );

            // Update current tag IDs and distances
            currentTagIDs.add(detection.id);
            tagDistances.put(detection.id, distance);

            // Update telemetry with detected tags
            telemetry.addData("Detected tag ID: " + detection.id, "Distance: " +
                    distance * 100);
            telemetry.addData("Detections :", detection.pose.y);
        }

        // Remove tags that are no longer detected
        Set<Integer> lostTags = new HashSet<>(tagDistances.keySet());
        lostTags.removeAll(currentTagIDs);
        for (Integer lostTagID : lostTags) {
            telemetry.addData("Tag ID: " + lostTagID, "Lost");
            tagDistances.remove(lostTagID); // Remove lost tags from distances
        }



        // Update telemetry
        telemetry.update();

        for (AprilTagDetection detection : detections)
        {
            Pose pose = poseFromTrapezoid(detection.corners, cameraMatrix, tagsizeX, tagsizeY);
            drawAxisMarker(input, tagsizeY / 2.0, 6, pose.rvec, pose.tvec, cameraMatrix);
            draw3dCubeMarker(input, tagsizeX, tagsizeX, tagsizeY, 5, pose.rvec, pose.tvec, cameraMatrix);
        }

        return input;
    }

    public void setDecimation(float decimation)
    {
        synchronized (decimationSync)
        {
            this.decimation = decimation;
            needToSetDecimation = true;
        }
    }

    public ArrayList<AprilTagDetection> getLatestDetections()
    {
        return detections;
    }

    public ArrayList<AprilTagDetection> getDetectionsUpdate()
    {
        synchronized (detectionsUpdateSync)
        {
            ArrayList<AprilTagDetection> ret = detectionsUpdate;
            detectionsUpdate = null;
            return ret;
        }
    }

    void constructMatrix()
    {
        //     Construct the camera matrix.
        //
        //      --         --
        //     | fx   0   cx |
        //     | 0    fy  cy |
        //     | 0    0   1  |
        //      --         --
        //

        cameraMatrix = new Mat(3,3, CvType.CV_32FC1);

        cameraMatrix.put(0,0, fx);
        cameraMatrix.put(0,1,0);
        cameraMatrix.put(0,2, cx);

        cameraMatrix.put(1,0,0);
        cameraMatrix.put(1,1,fy);
        cameraMatrix.put(1,2,cy);

        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2,1,0);
        cameraMatrix.put(2,2,1);
    }

    /**
     * Draw a 3D axis marker on a detection. (Similar to what Vuforia does)
     *
     * @param buf the RGB buffer on which to draw the marker
     * @param length the length of each of the marker 'poles'
     * @param rvec the rotation vector of the detection
     * @param tvec the translation vector of the detection
     * @param cameraMatrix the camera matrix used when finding the detection
     */
    void drawAxisMarker(Mat buf, double length, int thickness, Mat rvec, Mat tvec, Mat cameraMatrix)
    {
        // The points in 3D space we wish to project onto the 2D image plane.
        // The origin of the coordinate space is assumed to be in the center of the detection.
        MatOfPoint3f axis = new MatOfPoint3f(
                new Point3(0,0,0),
                new Point3(length,0,0),
                new Point3(0,length,0),
                new Point3(0,0,-length)
        );

        // Project those points
        MatOfPoint2f matProjectedPoints = new MatOfPoint2f();
        Calib3d.projectPoints(axis, rvec, tvec, cameraMatrix, new MatOfDouble(), matProjectedPoints);
        Point[] projectedPoints = matProjectedPoints.toArray();

        // Draw the marker!
        Imgproc.line(buf, projectedPoints[0], projectedPoints[1], red, thickness);
        Imgproc.line(buf, projectedPoints[0], projectedPoints[2], green, thickness);
        Imgproc.line(buf, projectedPoints[0], projectedPoints[3], blue, thickness);

        Imgproc.circle(buf, projectedPoints[0], thickness, white, -1);
    }

    void draw3dCubeMarker(Mat buf, double length, double tagWidth, double tagHeight, int thickness, Mat rvec, Mat tvec, Mat cameraMatrix)
    {
        //axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
        //       [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

        // The points in 3D space we wish to project onto the 2D image plane.
        // The origin of the coordinate space is assumed to be in the center of the detection.
        MatOfPoint3f axis = new MatOfPoint3f(
                new Point3(-tagWidth/2, tagHeight/2,0),
                new Point3( tagWidth/2, tagHeight/2,0),
                new Point3( tagWidth/2,-tagHeight/2,0),
                new Point3(-tagWidth/2,-tagHeight/2,0),
                new Point3(-tagWidth/2, tagHeight/2,-length),
                new Point3( tagWidth/2, tagHeight/2,-length),
                new Point3( tagWidth/2,-tagHeight/2,-length),
                new Point3(-tagWidth/2,-tagHeight/2,-length)
        );

        // Project those points
        MatOfPoint2f matProjectedPoints = new MatOfPoint2f();
        Calib3d.projectPoints(axis, rvec, tvec, cameraMatrix, new MatOfDouble(), matProjectedPoints);
        Point[] projectedPoints = matProjectedPoints.toArray();

        // Draw the bottom square
        for(int i = 0; i < 4; i++)
        {
            Imgproc.line(buf, projectedPoints[i], projectedPoints[(i+1)%4], green, thickness);
        }

        // Draw the top square
        for(int i = 4; i < 8; i++)
        {
            Imgproc.line(buf, projectedPoints[i], projectedPoints[4 + (i+1)%4], green, thickness);
        }

        // Draw the pillars
        for(int i = 0; i < 4; i++)
        {
            Imgproc.line(buf, projectedPoints[i], projectedPoints[i+4], green, thickness);
        }
    }

    /**
     * Used for finding the pose of the tag/marker
     */
    Pose poseFromTrapezoid(Point[] inPts, Mat cameraMatrix, double tagsizeX, double tagsizeY)
    {
        // Order points clockwise starting from point closest to origin
        Point[] sorted = sortPoints(inPts);

        MatOfPoint2f points = new MatOfPoint2f(sorted);
        MatOfPoint3f objPoints = new MatOfPoint3f(
                new Point3(-tagsizeX/2.0, tagsizeY/2.0, 0),
                new Point3(tagsizeX/2.0, tagsizeY/2.0, 0),
                new Point3(tagsizeX/2.0, -tagsizeY/2.0, 0),
                new Point3(-tagsizeX/2.0, -tagsizeY/2.0, 0)
        );

        Mat rvec = new Mat();
        Mat tvec = new Mat();
        Calib3d.solvePnP(objPoints, points, cameraMatrix, new MatOfDouble(), rvec, tvec, false);

        return new Pose(rvec, tvec);
    }

    /**
     * Sort points clockwise starting from point closest to origin
     * (reorders given array in place)
     */
    Point[] sortPoints(Point[] in)
    {
        ArrayList<Point> pts = new ArrayList<>();
        for (int i = 0; i < in.length; i++)
        {
            pts.add(in[i]);
        }

        // Sort by Y (to get top-down)
        pts.sort((lhs, rhs) ->
        {
            if (lhs.y < rhs.y)
                return -1;
            else if (lhs.y > rhs.y)
                return 1;
            else return 0;
        });

        Point[] out = {new Point(), new Point(), new Point(), new Point()};

        // Top left vs top right
        if (pts.get(0).x < pts.get(1).x)
        {
            out[0] = pts.get(0);
            out[1] = pts.get(1);
        }
        else
        {
            out[0] = pts.get(1);
            out[1] = pts.get(0);
        }

        // Bottom left vs bottom right
        if (pts.get(2).x < pts.get(3).x)
        {
            out[3] = pts.get(2);
            out[2] = pts.get(3);
        }
        else
        {
            out[3] = pts.get(3);
            out[2] = pts.get(2);
        }

        return out;
    }

    class Pose
    {
        public Mat rvec;
        public Mat tvec;

        public Pose(Mat rvec, Mat tvec)
        {
            this.rvec = rvec;
            this.tvec = tvec;
        }
    }
}
