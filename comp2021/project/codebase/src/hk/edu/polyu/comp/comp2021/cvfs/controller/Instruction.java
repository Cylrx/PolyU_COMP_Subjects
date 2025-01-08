package hk.edu.polyu.comp.comp2021.cvfs.controller;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.VirtualDisk;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.*;

import java.util.ArrayList;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.AssertUtils.*;


/**
 * Interface for all instructions.
 * Each instruction has two methods: execute and reverse.
 * execute():  used to execute the instruction. For normal execution and undoing
 * reverse():  used to reverse the instruction. For undoing only
 */
interface Instruction {
    /**
     * @throws InvalidFileException when the file is invalid
     * @throws FileExistsException when the file already exists
     * @throws FullDiskException when the disk is full
     * @throws InvalidTypeException when the type is invalid
     * @throws FileNotFoundException when the file is not found
     * @throws InvalidCriterionException when the criterion is invalid
     */
    void execute() throws InvalidFileException, FileExistsException, FullDiskException, InvalidTypeException, FileNotFoundException, InvalidCriterionException;

    /**
     * @throws FileExistsException when the file already exists
     * @throws FullDiskException when the disk is full
     * @throws FileNotFoundException when the file is not found
     * @throws InvalidCriterionException when the criterion is invalid
     * @throws InvalidFileException when the file name is invalid
     */
    void reverse() throws FileExistsException, FullDiskException, FileNotFoundException, InvalidCriterionException, InvalidFileException;
}

/**
 * Instruction to create a new document.
 */
class NewDoc implements Instruction {
    private final String name, content, type;
    private final VirtualDisk curDisk;
    private final Directory curDir;

    /**
     * @param curDisk current disk
     * @param curDir current directory
     * @param args arguments for the newDoc instruction
     * @throws InvalidCommandException when received number of arguments != 3
     */
    public NewDoc(VirtualDisk curDisk, Directory curDir, String[] args) throws InvalidCommandException {
        // [filename] [type] [content]
        assertArgumentCount("newDoc", args, 3);
        this.name = args[0]; // filename part
        this.type = args[1]; // type part
        this.content = args[2];
        this.curDisk = curDisk;
        this.curDir = curDir;
    }

    @Override
    public void execute() throws FileExistsException, InvalidFileException, FullDiskException, InvalidTypeException {
        Document doc = new Document(curDir, name, content, type);
        curDisk.newDoc(curDir, doc);
    }

    @Override
    public void reverse() throws FileNotFoundException {
        curDir.delete(name);
    }
}

/**
 * Instruction to create a new directory.
 */
class NewDir implements Instruction {
    private final String name;
    private final VirtualDisk curDisk;
    private final Directory curDir;

    /**
     * @param curDisk current disk
     * @param curDir current directory
     * @param args arguments for the newDir instruction
     * @throws InvalidCommandException when received number of arguments != 1
     */
    public NewDir(VirtualDisk curDisk, Directory curDir, String[] args) throws InvalidCommandException{
        assertArgumentCount("newDir", args, 1);
        this.name = args[0];
        this.curDisk = curDisk;
        this.curDir = curDir;
    }

    @Override
    public void execute() throws InvalidFileException, FullDiskException, FileExistsException {
        Directory dir = new Directory(name, curDir);
        curDisk.newDir(curDir, dir);
    }

    @Override
    public void reverse() throws FileNotFoundException {
        curDir.delete(name);
    }
}

/**
 * Instruction to delete a file
 * (can be either a document or a directory).
 */
class Delete implements Instruction {
    private final String name;
    private final Directory curDir;
    private final VirtualDisk curDisk;
    private Object file;

    /**
     * @param curDir current directory
     * @param curDisk current disk
     * @param args arguments for the delete instruction
     * @throws InvalidCommandException when received number of arguments != 1
     */
    public Delete(Directory curDir, VirtualDisk curDisk, String[] args) throws InvalidCommandException{
        assertArgumentCount("delete", args, 1);
        this.name = args[0];
        this.curDir = curDir;
        this.curDisk = curDisk;
    }

    @Override
    public void execute() throws FileNotFoundException {
        file = curDir.delete(name);
    }

    @Override
    public void reverse() {
        // parent folder must exist
        // because delete option of pDir should be undone first.
        try {
            if (file instanceof Directory) curDisk.newDir(curDir, (Directory) file);
            else curDisk.newDoc(curDir, (Document) file);
        } catch (FullDiskException | FileExistsException e) {
            throw new RuntimeException(e);
        }
    }
}

/**
 * Instruction to rename a file.
 */
class Rename implements Instruction {
    // rename oldName, newName

    private final String oldName, newName;
    private final Directory curDir;

    /**
     * @param curDir current directory
     * @param args arguments for the rename instruction
     * @throws InvalidCommandException when received number of arguments != 2
     */
    public Rename(Directory curDir, String[] args) throws InvalidCommandException{
        assertArgumentCount("rename", args, 2);
        this.oldName = args[0];
        this.newName = args[1];
        this.curDir = curDir;
    }

    @Override
    public void execute() throws FileNotFoundException, FileExistsException, InvalidFileException {
        curDir.rename(oldName, newName);
    }
    @Override
    public void reverse() {
        try {
            curDir.rename(newName, oldName);
        } catch (FileNotFoundException | FileExistsException | InvalidFileException e) {
            throw new RuntimeException("Impossible error: " + e);
        }
    }
}

/**
 * Instruction to create a new simple criterion.
 */
class NewSimpleCri implements Instruction {
    private final String crtName;
    private final VirtualDisk curDisk;
    private final Criterion crt;

    /**
     * @param curDisk current disk
     * @param args arguments for the newSimpleCri instruction
     * @throws InvalidCommandException when received number of arguments != 4
     * @throws InvalidCriterionException when the criterion arguments are invalid
     * @throws InvalidTypeException when attempted to create a "TypeCriterion" with invalid filetype argument
     * @throws InvalidValueException when attempted to create a "SizeCriterion" with invalid size argument (not a number, or greater than `Long` max value)
     */
    public NewSimpleCri(VirtualDisk curDisk, String[] args) throws InvalidCommandException, InvalidCriterionException, InvalidTypeException, InvalidValueException {
        // crtName, attrName, op, val
        assertArgumentCount("newSimpleCri", args, 4);
        assertValidCrtName(args[0]);
        this.curDisk = curDisk;
        this.crtName = args[0];
        String attrName = args[1];
        String op = args[2];
        String val = args[3];

        switch (attrName) {
            case "name": crt = new NameCriterion(crtName, op, val); break;
            case "type": crt = new TypeCriterion(crtName, op, val); break;
            case "size": crt = new SizeCriterion(crtName, op, val); break;
            default:
                throw Criterion.getInvalidCriterionException("attrName", "{name, type, size}", attrName);
        }
    }

    @Override
    public void execute() throws InvalidCriterionException {
        curDisk.newCrt(crtName, crt);
    }

    @Override
    public void reverse() throws InvalidCriterionException {
        curDisk.delCrt(crtName);
    }
}

/**
 * Instruction to create a new negation criterion.
 */
class NewNegation implements Instruction {
    private final String crtName;
    private final Criterion negCrt;
    private final VirtualDisk curDisk;

    /**
     * @param curDisk current disk
     * @param args arguments for the newNegation instruction
     * @throws InvalidCommandException when received number of arguments != 2
     * @throws InvalidCriterionException when the criterion arguments are invalid
     */
    public NewNegation(VirtualDisk curDisk, String[] args) throws InvalidCommandException, InvalidCriterionException {
        // crtName, crt
        assertArgumentCount("newNegation", args, 2);
        assertValidCrtName(args[0]);
        this.crtName = args[0];
        this.negCrt = new NegCriterion(crtName, curDisk.getCrt(args[1]));
        this.curDisk = curDisk;
    }

    @Override
    public void execute() throws InvalidCriterionException {
        curDisk.newCrt(crtName, negCrt);
    }

    @Override
    public void reverse() throws InvalidCriterionException {
        curDisk.delCrt(crtName);
    }
}

/**
 * Instruction to create a new binary criterion.
 */
class NewBinaryCri implements Instruction {
    private final VirtualDisk curDisk;
    private final String crtName;
    private final Criterion binCri;

    /**
     * @param curDisk current disk
     * @param args arguments for the newBinaryCri instruction
     * @throws InvalidCommandException when received number of arguments != 4
     * @throws InvalidCriterionException when the criterion arguments are invalid
     */
    public NewBinaryCri(VirtualDisk curDisk, String[] args) throws InvalidCommandException, InvalidCriterionException {
        // crtName, crt1, op, crt2
        assertArgumentCount("newBinaryCri", args, 4);
        assertValidCrtName(args[0]);
        this.curDisk = curDisk;
        this.crtName = args[0];
        Criterion c1 = curDisk.getCrt(args[1]);
        Criterion c2 = curDisk.getCrt(args[3]);
        this.binCri = new BinaryCriterion(crtName, args[2], c1, c2);
    }

    @Override
    public void execute() throws InvalidCriterionException {
        curDisk.newCrt(crtName, binCri);
    }

    @Override
    public void reverse() throws InvalidCriterionException {
        curDisk.delCrt(crtName);
    }
}

/**
 * ChangeDir class to change the current directory to a new directory.
 */
class ChangeDir implements Instruction{
    private final Controller control;
    private final Directory curDir;
    private final Directory newDir;
    private final boolean isReturn;

    /**
     * Constructor for ChangeDir class.
     * @param control the controller
     * @param args the arguments for the command
     * @throws InvalidCommandException when the number of arguments != 1
     * @throws FileNotFoundException when attempting to access parent directory at root directory
     */
    public ChangeDir (Controller control, String[] args) throws InvalidCommandException, FileNotFoundException {
        assertArgumentCount("changeDir", args, 1);
        this.control = control;
        this.curDir = control.getCurDir();
        this.isReturn = args[0].equals("..");
        this.newDir = (isReturn) ? curDir.getPa() : curDir.getDir(args[0]);

        if (newDir == null) {
            throw new FileNotFoundException("Cannot return: already at root");
        }
    }

    @Override
    public void execute() {
        control.setCurDir(newDir);
        ArrayList<String> cwd = control.getCwd();
        if (isReturn) cwd.remove(cwd.size() - 1);
        else cwd.add(newDir.getDirName());
    }

    @Override
    public void reverse() {
        control.setCurDir(curDir);
        ArrayList<String> cwd = control.getCwd();
        if (isReturn) cwd.add(curDir.getDirName());
        else cwd.remove(cwd.size() - 1);
    }
}

