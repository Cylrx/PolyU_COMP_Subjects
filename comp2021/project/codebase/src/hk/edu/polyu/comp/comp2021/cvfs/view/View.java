package hk.edu.polyu.comp.comp2021.cvfs.view;

import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.Criterion;
import hk.edu.polyu.comp.comp2021.cvfs.type.Pair;

import java.util.ArrayList;
import java.util.HashMap;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.StringUtils.repeatStr;

/**
 * The view class for the CVFS.
 */
public class View {

    private static final int MIN_STATS_LENGTH = 15;

    /**
     * Creates a new View object.
     */
    public View() {}

    /**
     * Prints an error message.
     * @param e the exception that caused the error
     */
    public void displayError(Exception e) { System.out.println("Error: " + e.getMessage()); }

    /**
     * Prints a message.
     * @param msg the message to be printed
     */
    public void displayMessage(String msg) { System.out.println(msg); }

    /**
     * Prints a new disk separator.
     */
    public void displayNewDisk() {
        System.out.print(System.lineSeparator());
        System.out.println("----- New Disk Created -----");
    }

    /**
     * Prints the file hierarchy as an ASCII file tree.
     * @param tree the file tree to be displayed. The tree is a list of objects, where each object is either a Document or a Pair<Directory, ArrayList<Object>>.
     */
    @SuppressWarnings("unchecked")
    public void displayFileTree (ArrayList<Object> tree) {
        long size = 0L, cnt = 0L;
        System.out.println(".");
        printASCIITree(tree, "");
        for (Object file: tree) {
            if (file instanceof Document) {
                Document doc = (Document) file;
                size += doc.getSize();
                cnt++;
            } else {
                Pair<Directory, ArrayList<Object>> pair = (Pair<Directory, ArrayList<Object>>) file;
                Directory dir = pair.getFirst();
                size += dir.getSize();
                cnt += dir.getCnt();
            }
        }
        printStats(0, size, cnt);
    }

    /**
     * Prints the list of document and directories (matched by the search criteria in the <code>Controller</code> class).
     * @param docs the list of matching documents. <code>Pair<String, Long></code> contains the full path and size of the document.
     * @param dirs the list of matching directories <code>Pair<String, Long></code> contains the full path and size of the directory.
     */
    public void displayFileList (ArrayList<Pair<String, Long>> docs, ArrayList<Pair<String, Long>> dirs) {
        long size = 0L, cnt = docs.size() + dirs.size();
        for (Pair<String, Long>p: docs) size += p.getSecond();
        for (Pair<String, Long>p: dirs) size += p.getSecond();

        int maxLen = MIN_STATS_LENGTH;
        for (Pair<String, Long>p: docs) maxLen = Math.max(maxLen, p.getFirst().length());
        for (Pair<String, Long>p: dirs) maxLen = Math.max(maxLen, p.getFirst().length());

        String template = "%-" + maxLen + "s %d%n";
        System.out.println("\nMatching Documents:");
        for (Pair<String, Long>p: docs) System.out.printf(template, p.getFirst(), p.getSecond());
        System.out.println("\nMatching Directories:");
        for (Pair<String, Long>p: dirs) System.out.printf(template, p.getFirst(), p.getSecond());
        printStats(maxLen, size, cnt);
    }

    /**
     * Prints the list of criteria in the virtual disk.
     * @param cMap the hash map of criteria in the virtual disk.
     */
    public void displayCriteria(HashMap<String, Criterion> cMap) {
        for (Criterion crt: cMap.values()) {
            String expr = crt.getExpr();
            String name = crt.getCrtName();
            System.out.println(name + "  " + expr);
        }
    }

    /**
     * Prints the prompt for the user to enter a command.
     * @param cwd the current working directory as a list of directory names.
     */
    public void displayPrompt (ArrayList<String> cwd) {
        StringBuilder prompt = new StringBuilder();
        if (cwd.size() != 1 || !cwd.get(0).equals("*")) {
            prompt.append("\n~/");
            for (String dir : cwd) prompt.append(dir).append("/");
        }
        prompt.append("\n>>>  ");
        System.out.print(prompt);
    }

    @SuppressWarnings("unchecked")
    private void printASCIITree (ArrayList<Object> tree, String curPrefix) {
        int n = tree.size();
        for (int i = 0; i < n; i++) {
            String curLine = curPrefix;
            String nextPrefix = curPrefix;

            // if last element of the tree, use corner & no need vertical line for prefix
            if ((i != n - 1)) {
                curLine += "├─";
                nextPrefix += "│\t";
            } else {
                curLine += "└─";
                nextPrefix += "\t";
            }

            Object file = tree.get(i);

            if (file instanceof Pair) {
                Pair<Directory, ArrayList<Object>> pair = (Pair<Directory, ArrayList<Object>>) file;
                ArrayList<Object> subTree = pair.getSecond();
                Directory dir = pair.getFirst();
                String name = dir.getDirName();
                long size = dir.getSize();
                System.out.println(curLine + getDirEntry(name, size));
                printASCIITree(subTree, nextPrefix);
            } else {
                Document doc = (Document) file;
                String name = doc.getDocName();
                String type = doc.getType().toStrType();
                long size = doc.getSize();
                System.out.println(curLine + getDocEntry(name, type, size));
            }
        }
    }

    private void printStats(int maxLen, long totalSize, long totalFiles) {
        maxLen = Math.max(maxLen, MIN_STATS_LENGTH);
        String template = "%-" + maxLen + "s %-15s%n";
        System.out.println(repeatStr('-', maxLen + 10));
        System.out.printf(template, "Total Files", "Total Size");
        System.out.printf(template, totalFiles, totalSize);
    }

    private String getDocEntry(String name, String type, long size) {
        String filename = name + "." + type;
        return String.format("%-10s size: %d", filename, size);
    }

    private String getDirEntry(String name, long size) {
        name += "/";
        return String.format("%-10s size: %d", name, size);
    }

}
