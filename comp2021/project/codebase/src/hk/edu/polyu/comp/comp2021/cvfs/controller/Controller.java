package hk.edu.polyu.comp.comp2021.cvfs.controller;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.VirtualDisk;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.Criterion;
import hk.edu.polyu.comp.comp2021.cvfs.type.Pair;
import hk.edu.polyu.comp.comp2021.cvfs.utils.SerializeUtils;
import hk.edu.polyu.comp.comp2021.cvfs.view.View;

import java.util.ArrayList;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.AssertUtils.*;
import static hk.edu.polyu.comp.comp2021.cvfs.utils.StringUtils.splitStr;
import static hk.edu.polyu.comp.comp2021.cvfs.utils.TypeUtils.toNumber;

/**
 *  Controller class of the MVC pattern
 *  Handles user input and executes instructions by manipulating the model and sending results to the view
 */
public class Controller {
    private static final long MAX_DISK_SIZE = 1_000_000_000;
    private Directory curDir = null;
    private VirtualDisk curDisk = null;
    private final View view;
    private final History history;
    private final ArrayList<String> cwd;
    private boolean isRunning;


    /**
     * @param view view object to link to
     */
    public Controller(View view){
        this.view = view;
        this.history = new History();
        this.cwd = new ArrayList<>();
        this.isRunning = true;
        this.cwd.add("*"); // no disk loaded
        view.displayPrompt(cwd);
    }

    /**
     * @return the current loaded disk
     */
    protected Directory getCurDir() {
        return curDir;
    }

    /**
     * @return the current working directory of the current loaded disk
     */
    protected ArrayList<String> getCwd () {
        return cwd;
    }

    /**
     * Sets the current working directory of the current loaded disk
     * @param newDir the new directory to set as the current working directory
     */
    protected void setCurDir(Directory newDir) {
        assert newDir != null;
        curDir = newDir;
    }

    /**
     * This method takes in a user input command and executes it.
     * The effects are immediately reflect on the model and the view, without the need for any additional method calls.
     * This method is essentially the entry point for instruction execution.
     *
     * @param str user input command in a single line
     * @return true if the program is still running, false if the program is to be terminated
     */
    public boolean runInstruction (String str) {
        // if inst not null, instruction can be redone / undone
        // if inst null, instruction has already been executed
        // Parse Instruction
        try {
            Instruction inst = parseInstruction(str);
            if (inst != null) history.run(inst);
        } catch (CVFSException e){
            view.displayError(e);
        }

        if (isRunning) view.displayPrompt(cwd);
        return isRunning;
    }

    private Instruction parseInstruction (String str) throws
            CVFSException {
        String[] tokens = splitStr(str);
        if (tokens.length == 0) return null;
        String[] args = parseArgs(tokens);
        String cmd = tokens[0];
        return parseInstruction(cmd, args);
    }

    private static String[] parseArgs (String[] arr) {
        int length = arr.length - 1;
        String[] args = new String[length];
        System.arraycopy(arr, 1, args, 0, length);
        return args;
    }

    private Instruction parseInstruction (String cmd, String[] args) throws CVFSException{
        if (curDisk == null) {
            switch (cmd) {
                case "newDisk": case "quit": case "load": break;
                default: throw new InvalidActionException("Cannot run " + cmd + ": no disk loaded");
            }
        }
        switch (cmd) {
            // Commands that can be undone and redone
            case "newDoc": return new NewDoc(curDisk, curDir, args);
            case "newDir": return new NewDir(curDisk, curDir, args);
            case "delete": return new Delete(curDir, curDisk, args);
            case "rename": return new Rename(curDir, args);
            case "newSimpleCri": return new NewSimpleCri(curDisk, args);
            case "newNegation": return new NewNegation(curDisk, args);
            case "newBinaryCri": return new NewBinaryCri(curDisk, args);
            case "changeDir": return new ChangeDir(this, args);

            // Commands that cannot be undone or redone
            case "search": search(args); break;
            case "rSearch": rSearch(args); break;
            case "newDisk": newDisk(args); break;
            case "rList": rList(args); break;
            case "list": list(args); break;
            case "quit": quit(args); break;
            case "undo": undo(args); break;
            case "redo": redo(args); break;
            case "save": save(args); break;
            case "load": load(args); break;
            case "printAllCriteria": printAllCriteria(); break;

            default: throw new InvalidCommandException("command not found: " + cmd);
        }
        return null;
    }

    private void quit(String[] args) throws InvalidCommandException {
        assertArgumentCount("quit", args, 0);
        isRunning = false;
    }

    private void undo(String[] args) throws CVFSException {
        assertArgumentCount("undo", args, 0);
        history.undo();
    }

    private void redo(String[] args) throws CVFSException{
        assertArgumentCount("redo", args, 0);
        history.redo();
    }

    private void newDisk(String[] args) throws InvalidCommandException, InvalidValueException {
        assertArgumentCount("newDisk", args, 1);
        assertNumber("disk size", args[0]);
        assertLimit("disk size", args[0], MAX_DISK_SIZE);
        curDisk = new VirtualDisk(toNumber(args[0]));
        curDir = curDisk.getRoot();
        history.clear();
        cwd.clear();
        view.displayNewDisk();
    }

    private void rList(String[] args) throws InvalidCommandException {
        assertArgumentCount("rList", args, 0);
        view.displayFileTree(treeDfs(curDir, true));
    }

    private void list(String[] args) throws InvalidCommandException {
        assertArgumentCount("list", args, 0);
        view.displayFileTree(treeDfs(curDir, false));
    }

    private void rSearch(String[] args) throws InvalidCommandException, InvalidCriterionException {
        assertArgumentCount("rSearch", args, 1);
        Criterion crt = curDisk.getCrt(args[0]);
        ArrayList<Pair<String, Object>> docs = new ArrayList<>();
        ArrayList<Pair<String, Object>> dirs = new ArrayList<>();
        listDfs(curDir, docs, dirs, "/", true);
        view.displayFileList(filterList(docs, crt), filterList(dirs, crt));
    }

    private void search(String[] args) throws InvalidCommandException, InvalidCriterionException {
        assertArgumentCount("search", args, 1);
        Criterion crt = curDisk.getCrt(args[0]);
        ArrayList<Pair<String, Object>> docs = new ArrayList<>();
        ArrayList<Pair<String, Object>> dirs = new ArrayList<>();
        listDfs(curDir, docs, dirs, "/", false);
        view.displayFileList(filterList(docs, crt), filterList(dirs, crt));
    }

    private void save(String[] args) throws InvalidCommandException, InvalidIOException {
        assertArgumentCount("save", args, 1);
        SerializeUtils.serialize(curDisk, args[0]);
        view.displayMessage("save: saved serialized disk to " + args[0]);
    }

    private void load(String[] args) throws InvalidCommandException, InvalidIOException {
        assertArgumentCount("save", args, 1);
        curDisk = (VirtualDisk) SerializeUtils.deserialize(args[0]);
        assert curDisk != null;
        curDir = curDisk.getRoot();
        view.displayMessage("load: loaded serialized disk from " + args[0]);
        cwd.clear();
    }

    private void printAllCriteria () {
        view.displayCriteria(curDisk.getCMap());
    }

    // Returned ArrayList captures the file tree for list methods
    private ArrayList<Object> treeDfs(Directory curDfsDir, boolean recursive) {
        ArrayList<Object> res = new ArrayList<>();
        for (Directory dir: curDfsDir.getSubDirs().values()) {
            ArrayList<Object> subDirFiles;

            if (recursive) subDirFiles = treeDfs(dir, true);
            else subDirFiles =new ArrayList<>();

            Pair<Directory, ArrayList<Object>> subDir = new Pair<>(dir, subDirFiles);
            res.add(subDir);
        }
        res.addAll(curDfsDir.getSubDocs().values());
        return res;
    }

    // Returns flat ArrayList (no file tree structure captured) of all subdirectories and documents for rSearch method
    private void listDfs(Directory curDfsDir, ArrayList<Pair<String, Object>> docs, ArrayList<Pair<String, Object>> dirs, String cwd, boolean recursive) {
        for (Document doc: curDfsDir.getSubDocs().values()) {
            String docPath = String.format("%s%s.%s", cwd, doc.getDocName(), doc.getType().toStrType());
            docs.add(new Pair<>(docPath, doc));
        }
        for (Directory dir: curDfsDir.getSubDirs().values()) {
            String dirPath = String.format("%s%s/", cwd, dir.getDirName());
            dirs.add(new Pair<>(dirPath, dir));
            if (recursive) listDfs(dir, docs, dirs, dirPath, true);
        }
    }

    private ArrayList<Pair<String, Long>> filterList(ArrayList<Pair<String, Object>> entries, Criterion crt) {
        ArrayList<Pair<String, Long>> res = new ArrayList<>();

        for (Pair<String, Object> entry: entries) {
            Object file = entry.getSecond();
            String path = entry.getFirst();

            if (file instanceof Document) {
                Document doc = (Document) file;
                if (crt.eval(doc)) {
                    res.add(new Pair<>(path, doc.getSize()));
                }
            } else {
                Directory dir = (Directory) file;
                if (crt.eval(dir)) {
                    res.add(new Pair<>(path, dir.getSize()));
                }
            }
        }
        return res;
    }
}
